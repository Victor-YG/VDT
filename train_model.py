import os
import time
import argparse
import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as nn_func

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import albumentations as alb

from core.dcnet import DCnet
from utils.loss import MAE_loss, MSE_loss
from utils.data.kitti import KITTI


# hyperparameters
LEARNING_RATE = 1e-3
CROP_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))


# transform
transform = alb.Compose(
    [
        alb.RandomCrop(CROP_SIZE, CROP_SIZE, True, 1),
    ],
    additional_targets = {
        "img_d" : "image",
        "img_t" : "image"
    }
)


def get_time_stamp():
    time = datetime.datetime.now()
    return time.strftime(r"%Y%m%d_%H%M%S")


def save_checkpoint(model, optimizer, file_path=None):
    if file_path == None:
        try:
            model_name = model.name()
        except:
            model_name = "random"
        file_path = "./training/{0}_{1}.pth.tar".format(model_name, get_time_stamp())
    
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["optimizer"] = optimizer.state_dict()
    
    torch.save(checkpoint, file_path)
    print("Saved checkpoint...\n")


def load_checkpoint(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("Loaded checkpoint from file '{}'".format(file_path))


def prepare_completion_input_and_label(image_paths):
    '''load images and apply transformation'''

    # transform images
    img_paths_c = image_paths['c']
    img_paths_d = image_paths['d']
    img_paths_t = image_paths['t']
    imgs_c = []
    imgs_d = []
    imgs_t = []

    for (c, d, t) in zip(img_paths_c, img_paths_d, img_paths_t):
        img_c = np.array(Image.open(c))
        img_d = np.array(Image.open(d))
        img_t = np.array(Image.open(t))

        img_trans = transform(image=img_c, img_d=img_d, img_t=img_t)
        img_trans_c = img_trans["image"]
        img_trans_d = img_trans["img_d"]
        img_trans_t = img_trans["img_t"]

        img_trans_c = np.divide(img_trans_c, 255.0)
        # img_trans_d = np.divide(img_trans_d, 256.0)
        # img_trans_t = np.divide(img_trans_t, 256.0)
        img_trans_d = np.multiply(256.0, np.reciprocal(img_trans_d)) # np.divide(256.0, img_trans_d)
        img_trans_t = np.multiply(256.0, np.reciprocal(img_trans_t)) # np.divide(256.0, img_trans_t)
        
        imgs_c.append(img_trans_c)
        imgs_d.append(img_trans_d)
        imgs_t.append(img_trans_t)

    # convert to tensor
    tensor_c = torch.tensor(imgs_c, dtype=torch.float32)
    tensor_d = torch.tensor(imgs_d, dtype=torch.float32)
    tensor_t = torch.tensor(imgs_t, dtype=torch.float32)

    # adjust dimension and concat
    tensor_c = torch.einsum("nhwc->nchw", tensor_c)
    tensor_d = torch.reshape(tensor_d, [tensor_d.shape[0], -1, tensor_d.shape[1], tensor_d.shape[2]])
    tensor_t = torch.reshape(tensor_t, [tensor_t.shape[0], -1, tensor_t.shape[1], tensor_t.shape[2]])

    return tensor_c, tensor_d, tensor_t


def train_completion_model(model, dataloader, loss_func, epoches, device="cpu"):
    if device == "cuda":
        torch.cuda.empty_cache()

    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()

    training_start_time = time.monotonic()

    for i in range(epoches):
        print("epoch {}: ".format(i))
        print("-------------------------------------------------")
        losses = []

        for batch_idx, image_paths in enumerate(dataloader):
            start_time = time.monotonic()

            # data preparation
            tensor_c, tensor_d, tensor_t = prepare_completion_input_and_label(image_paths)
            tensor_c = tensor_c.to(device)
            tensor_d = tensor_d.to(device)
            tensor_t = tensor_t.to(device)

            # forward
            tensor_output = model(tensor_c, tensor_d)
            loss = loss_func(tensor_output, tensor_t)
            losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.monotonic()
            print("batch {} took {:.1f} secs; loss = {:.2f}".format(batch_idx + 1, end_time - start_time, loss.item()))

        print("average loss at epoch {} = {:.2f}\n".format(i + 1, sum(losses) / len(losses)))

        # if i % 1 == 0:
        save_checkpoint(model, optimizer)

    training_end_time = time.monotonic()
    print("Training of {} epoches took {:.1f} secs.".format(epoches, training_end_time - training_start_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-kitti_folder", help="Folder containing KITTI dataset.", default=r"..\..\datasets\DCNE", required=False)
    parser.add_argument("-batch_size", type=int, default=32, required=False)
    parser.add_argument("-epoches", type=int, default=2, required=False)
    parser.add_argument("-checkpoint", type=str, help="Checkpoint file path.", default="", required=False)
    args = parser.parse_args()

    # check arguments
    if os.path.exists(args.kitti_folder) == False:
        exit("Path to kitti_folder doesn't exist.")

    if args.checkpoint == "":
        print("Training from scratch? [y/n]")

        while True:
            user_input = input()
            if user_input == "y" or user_input == "Y":
                break
            if user_input == "n" or user_input == "N":
                print("Enter path to the checkpoint: ")
                user_input = input()
                if os.path.exists(user_input):
                    args.checkpoint = user_input
                    break
                else:
                    exit("User entered checkpoint file doesn't exist.")

    # load data
    kitti_dataset = KITTI(args.kitti_folder, transform, mode="monodepth", split="train")
    kitti_dataloader = DataLoader(dataset=kitti_dataset, batch_size=args.batch_size)

    model = DCnet().to(DEVICE)
    mse_loss = MAE_loss()

    if args.checkpoint != "":
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])

    # train model
    train_completion_model(model, kitti_dataloader, loss_func=mse_loss, epoches=args.epoches, device=DEVICE)


if __name__ == "__main__":
    main()