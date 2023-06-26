import os
import time
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from core.PSMNet.stackhourglass import PSMNet


# check hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO]: Using device '{}'".format(DEVICE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Checkpoint file path.", required=True)
    parser.add_argument('--img_l', help="Left image of stereo pair", required=True)
    parser.add_argument('--img_r', help="Right image of stereo pair", required=True)
    parser.add_argument("--maxdisp", help="Maximum disparity allowed", type=int, default=192, required=False)
    args = parser.parse_args()

    # check arguments
    if os.path.exists(args.checkpoint) == False:
        exit("[FAIL]: Checkpoint file doesn't exist.")

    torch.manual_seed(1)
    if DEVICE == "cuda": torch.cuda.manual_seed(1)

    # create model
    model = PSMNet(args.maxdisp)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()
    model.eval()

    # load checkpoint
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict["state_dict"])
    print("[INFO]: Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    normal_mean_var = {}
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load input images
    img_l = Image.open(args.img_l).convert("RGB")
    img_r = Image.open(args.img_r).convert("RGB")
    img_l = test_transform(img_l)
    img_r = test_transform(img_r)

    # pad to width and hight to 16 times
    if img_l.shape[1] % 16 != 0:
        times = img_l.shape[1] // 16
        top_pad = (times + 1) * 16 - img_l.shape[1]
    else:
        top_pad = 0

    if img_l.shape[2] % 16 != 0:
        times = img_l.shape[2] // 16
        right_pad = (times + 1) * 16 - img_l.shape[2]
    else:
        right_pad = 0

    img_l = F.pad(img_l, (0, right_pad, top_pad, 0)).unsqueeze(0)
    img_r = F.pad(img_r, (0, right_pad, top_pad, 0)).unsqueeze(0)
    if DEVICE == "cuda": img_l, img_r = img_l.cuda(), img_r.cuda()

    # run model
    start_time = time.monotonic()
    with torch.no_grad(): img_p = model(img_l, img_r)
    end_time = time.monotonic()

    img_p = torch.squeeze(img_p)
    img_d = img_p.data.cpu().numpy()

    print("[INFO]: Time spent is {:.3f} secs.".format(end_time - start_time))

    # remove padding
    if top_pad !=0 and right_pad != 0:
        img_d = img_d[top_pad : , : -right_pad]
    elif top_pad ==0 and right_pad != 0:
        img_d = img_d[ : , : -right_pad]
    elif top_pad !=0 and right_pad == 0:
        img_d = img_d[top_pad : , : ]
    else:
        img_d = img_d

    # save output image
    img_d = (img_d * 256).astype("uint16")
    img_d = Image.fromarray(img_d)
    img_d.save("output_disparity.png")


if __name__ == '__main__':
   main()
