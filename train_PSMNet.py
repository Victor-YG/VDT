import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_time_stamp
from utils.sceneflow import sceneflow_loader
from core.PSMNet.stackhourglass import PSMNet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO]: Using device '{}'.".format(DEVICE))


def train_PSMNet(model, checkpoint, dataloader_train, epoches, max_disparity=192, device="cpu"):
    '''Train PSMNet with provided data'''

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizer.zero_grad()

    # prep random seed and model
    torch.manual_seed(1)
    if device == "cuda":
        torch.cuda.manual_seed(1)
        torch.cuda.empty_cache()
        model = nn.DataParallel(model)
        model.cuda()

    # load checkpoint
    start_epoch = 0
    if checkpoint != None:
        checkpoint_dict = torch.load(checkpoint)
        start_epoch = checkpoint_dict["epoch"] + 1
        model.load_state_dict(checkpoint_dict["state_dict"])

    model.train(True)
    training_start_time = time.monotonic()

    for i in range(start_epoch, start_epoch + epoches):
        print("[INFO]: epoch {}: ".format(i))
        print("-------------------------------------------------")
        losses = []

        for batch_idx, [img_l, img_r, img_d] in enumerate(dataloader_train):
            start_time = time.monotonic()

            # data preparation
            if device == "cuda":
                img_l, img_r, img_d = img_l.cuda(), img_r.cuda(), img_d.cuda()

            mask = img_d < max_disparity
            mask.detach_()

            # run training
            output1, output2, output3 = model(img_l, img_r)
            output1 = torch.squeeze(output1, 1)
            output2 = torch.squeeze(output2, 1)
            output3 = torch.squeeze(output3, 1)
            loss = 0.5 * F.smooth_l1_loss(output1[mask], img_d[mask], size_average=True)\
                 + 0.7 * F.smooth_l1_loss(output2[mask], img_d[mask], size_average=True)\
                 + 1.0 * F.smooth_l1_loss(output3[mask], img_d[mask], size_average=True)

            loss.backward()
            losses.append(loss)

            optimizer.step()
            optimizer.zero_grad()

            end_time = time.monotonic()
            print("[INFO]: Batch {} took {:.1f} secs; loss = {:.2f}".format(batch_idx + 1, end_time - start_time, loss))

        print("[INFO]: Average loss at epoch {} = {:.2f}.\n".format(i + 1, sum(losses) / len(losses)))

        # save checkpoint
        file_path = "./training/PSMNet_{0}.tar".format(i)
        torch.save({'epoch': i, "state_dict" : model.state_dict()}, file_path)

    training_end_time = time.monotonic()
    print("[INFO]: Trained for {} epoches took {:.1f} secs in total.".format(epoches, training_end_time - training_start_time))


def test_PSMNet(model, dataloader_test, max_disparity=192, device="cpu"):
    '''Test PSMNet with provided data'''

    model.eval()
    lossess = []

    for batch_idx, [img_l, img_r, img_d] in enumerate(dataloader_test):
        if device == "cuda":
            img_l, img_r, img_d = img_l.cuda(), img_r.cuda(), img_d.cuda()
        mask = img_d < max_disparity

        # pad input
        if img_l.shape[2] % 16 != 0:
            times = img_l.shape[2] // 16
            top_pad = (times + 1) * 16 - img_l.shape[2]
        else:
            top_pad = 0

        if img_l.shape[3] % 16 != 0:
            times = img_l.shape[3] // 16
            right_pad = (times + 1) * 16 - img_l.shape[3]
        else:
            right_pad = 0

        img_l = F.pad(img_l, (0, right_pad, top_pad, 0))
        img_r = F.pad(img_r, (0, right_pad, top_pad, 0))

        # inference
        with torch.no_grad():
            output3 = model(img_l, img_r)
            output3 = torch.squeeze(output3)

        # remove padding
        # (only requird to remove top padding as the mask later will remove the right padding)
        img_p = output3[ : , top_pad : , : ] if top_pad != 0 else output3

        # compute loss
        loss = F.l1_loss(img_p[mask], img_d[mask]) if len(img_d[mask]) != 0 else 0
        loss = loss.data.cpu()

        print("[INFO]: Batch {} test loss = {:.3f}".format(batch_idx, loss))
        lossess.append(loss)

    print("[INFO]: Average test loss = {:.3f}".format(sum(lossess) / len(lossess)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder containing training/testing data", required=True)
    parser.add_argument("--dataset", help="Name of the dataset", default="sceneflow", required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--epoches", type=int, default=1, required=False)
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path.", default=None, required=False)
    parser.add_argument("--w", help="Width of image crop", type=int, default=512, required=False)
    parser.add_argument("--h", help="Height of image crop", type=int, default=256, required=False)
    parser.add_argument("--maxdisp", help="Maximum disparity allowed", type=int, default=192, required=False)
    args = parser.parse_args()

    # check arguments
    if os.path.exists(args.folder) == False:
        exit("[FAIL]: Dataset folder doesn't exist.")

    if args.checkpoint != None:
        print("[INFO]: Continue training from checkpoint '{}'.".format(args.checkpoint))
        if os.path.exists(args.checkpoint) == False:
            exit("[FAIL]: Checkpoint '{}' doesn't exist!".format(args.checkpoint))
    else:
        print("[INFO]: Training from scratch? [y/n]")
        while True:
            user_input = input()
            if user_input == "y" or user_input == "Y":
                break
            if user_input == "n" or user_input == "N":
                exit("[FAIL]: Specify the checkpoint with '--checkpoint'")

    # dataloader
    sceneflow_train = sceneflow_loader(root=args.folder, split="train", crop_size=[args.w, args.h])
    dataloader_train = torch.utils.data.DataLoader(dataset=sceneflow_train, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=False)

    sceneflow_test = sceneflow_loader(root=args.folder, split="test", crop_size=[args.w, args.h])
    dataloader_test = torch.utils.data.DataLoader(dataset=sceneflow_test, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    # create model
    model = PSMNet(args.maxdisp).to(DEVICE)
    print("[INFO]: Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    # train model
    train_PSMNet(model, args.checkpoint, dataloader_train, epoches=args.epoches, max_disparity=args.maxdisp, device=DEVICE)

    # test model
    test_PSMNet(model, dataloader_test, device=DEVICE)


if __name__ == "__main__":
    main()
