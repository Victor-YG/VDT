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

import params
from VDT import VDT
from utils.loss import MAE_loss, MSE_loss
from utils.data.kitti import KITTI


# hyperparameters
CROP_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))


# transform
transform = alb.Compose(
    [
        alb.RandomCrop(CROP_SIZE, CROP_SIZE, True, 1),
    ],
    additional_targets = {
        "img_r" : "image",
        "img_T" : "image"
    }
)


def prepare_stereo_matching_input_and_label(image_paths, device="cpu"):
    '''load images and apply transformation'''

    # transform images
    img_paths_l = image_paths['l_c']
    img_paths_r = image_paths['r_c']
    img_paths_T = image_paths['l_t']
    imgs_l = []
    imgs_r = []
    imgs_T = []

    for (l, r, T) in zip(img_paths_l, img_paths_r, img_paths_T):
        img_l = np.array(Image.open(l))
        img_r = np.array(Image.open(r))
        img_T = np.array(Image.open(T))

        img_trans = transform(image=img_l, img_r=img_r, img_T=img_T)
        img_trans_l = img_trans["image"]
        img_trans_r = img_trans["img_r"]
        img_trans_T = img_trans["img_T"]

        img_trans_l = np.divide(img_trans_l, 255.0)
        img_trans_r = np.divide(img_trans_r, 255.0)
        img_trans_T = np.divide(img_trans_T, 256.0)
        
        imgs_l.append(img_trans_l)
        imgs_r.append(img_trans_r)
        imgs_T.append(img_trans_T)

    # convert to tensor
    tensor_l = torch.tensor(imgs_l, dtype=torch.float32)
    tensor_r = torch.tensor(imgs_r, dtype=torch.float32)
    tensor_T = torch.tensor(imgs_T, dtype=torch.float32)

    # adjust dimension and concat
    tensor_l = torch.einsum("nhwc->nchw", tensor_l)
    tensor_r = torch.reshape(tensor_r, [tensor_r.shape[0], -1, tensor_r.shape[1], tensor_r.shape[2]])
    tensor_T = torch.reshape(tensor_T, [tensor_T.shape[0], -1, tensor_T.shape[1], tensor_T.shape[2]])

    return tensor_l.to(device), tensor_r.to(device), None, tensor_T.to(device)


def train_VDT(model, dataloader, epoches, device="cpu"):
    if device == "cuda":
        torch.cuda.empty_cache()

    model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
    optimizer.zero_grad()

    training_start_time = time.monotonic()

    for i in range(epoches):
        print("epoch {}: ".format(i))
        print("-------------------------------------------------")
        losses = []

        for batch_idx, image_paths in enumerate(dataloader):
            start_time = time.monotonic()

            # data preparation
            tensor_l, tensor_r, tensor_t, tensor_T = prepare_stereo_matching_input_and_label(image_paths, device)

            # run training
            loss = model.train_networks(tensor_l, tensor_r, tensor_t, tensor_T)
            losses.append(loss)

            end_time = time.monotonic()
            print("batch {} took {:.1f} secs; loss = {:.2f}".format(batch_idx + 1, end_time - start_time, loss))

        print("average loss at epoch {} = {:.2f}\n".format(i + 1, sum(losses) / len(losses)))

        # if i % 1 == 0:
        model.save_checkpoint()

    training_end_time = time.monotonic()
    print("Training of {} epoches took {:.1f} secs.".format(epoches, training_end_time - training_start_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-kitti_folder", help="Folder containing KITTI dataset.", default=r"..\..\datasets\kitti\depth", required=False)
    parser.add_argument("-batch_size", type=int, default=32, required=False)
    parser.add_argument("-epoches", type=int, default=10, required=False)
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
    kitti_dataset = KITTI(args.kitti_folder, transform, mode="stereodepth", split="train")
    kitti_dataloader = DataLoader(dataset=kitti_dataset, batch_size=args.batch_size)

    # create model
    model = VDT().to(DEVICE)

    # load checkpoint
    if args.checkpoint != "":
        model.load_checkpoint(args.checkpoint)

    # train model
    train_VDT(model, kitti_dataloader, epoches=args.epoches, device=DEVICE)


if __name__ == "__main__":
    main()