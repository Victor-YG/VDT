import os
import random
import argparse

import numpy as np
from PIL import Image

import torch.utils.data as data
from torchvision import transforms
import torch.functional as F

from general import read_pfm
from transform import normalize


class sceneflow_loader(data.Dataset):
    '''this object manages the scenflow dataset'''

    # potential split: "train", "test"
    # data tuple: "left - l", "right - r", "disparity - d"
    # dataset folder is assumed to be aranged in the following way
    #  - ${root}/flyingthings3d/frames_finalpass/TRAIN/A/0000/left
    #  - ${root}/flyingthings3d/frames_disparity/TRAIN/A/0000/left

    def __init__(self, root, split="train", crop_size=None):
        self.split = split
        self.crop_size = crop_size

        self.images_l = []
        self.images_r = []
        self.images_d = []

        labels = ['A', 'B', 'C']
        split_name = "TRAIN" if split == "train" else "TEST"

        #################
        # flyingthing3d #
        #################
        finalpass_folder = os.path.join(root, "flyingthings3d")
        finalpass_folder = os.path.join(finalpass_folder, "frames_finalpass")
        finalpass_folder = os.path.join(finalpass_folder, split_name)

        # get all left images
        for label in labels:
            finalpass_sub_folder = os.path.join(finalpass_folder, label)

            for batch_id in os.listdir(finalpass_sub_folder):
                batch_folder = os.path.join(finalpass_sub_folder, batch_id)
                left_folder = os.path.join(batch_folder, "left")
                images_l = os.listdir(left_folder)
                for img_l in images_l:
                    self.images_l.append(os.path.join(left_folder, img_l))

        for img_l in self.images_l:
            # get all right images
            self.images_r.append(img_l.replace("/left/", "/right/"))

            # get all disparity iamges
            img_d = img_l.replace("/frames_finalpass/", "/frames_disparity/")
            img_d = img_d.replace(".png", ".pfm")
            self.images_d.append(img_d)

        print("[INFO]: sceneflow_loader: Loaded {} set of data.".format(len(self.images_l)))

        # # for debug
        # with open("list_of_sceneflow_iamges.txt", "w") as f:
        #     for img_l in self.images_l:
        #         print(img_l, file=f, end="\n")


    def __getitem__(self, index):
        img_path_l = self.images_l[index]
        img_path_r = self.images_r[index]
        img_path_d = self.images_d[index]

        # load images
        img_l = Image.open(img_path_l).convert('RGB')
        img_r = Image.open(img_path_r).convert('RGB')
        img_d, scale_d = read_pfm(img_path_d)
        img_d = np.ascontiguousarray(img_d, dtype=np.float32)

        if self.split == "train" and self.crop_size is not None:
            # crop images
            w, h = img_l.size
            tw, th = self.crop_size

            x0 = random.randint(0, w - tw)
            y0 = random.randint(0, h - th)

            img_l = img_l.crop((x0, y0, x0 + tw, y0 + th))
            img_r = img_r.crop((x0, y0, x0 + tw, y0 + th))
            img_d = img_d[y0 : y0 + th, x0 : x0 + tw]

        # nornalize rgb image
        transform_normalize = normalize()
        img_l = transform_normalize(img_l)
        img_r = transform_normalize(img_r)

        return img_l, img_r, img_d


    def __len__(self):
        return len(self.images_l)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder containing sceneflow dataset", required=True)
    parser.add_argument("--split", help="What type of data (split) to load", default="train", required=False)
    parser.add_argument("--w", help="Width of image crop", type=int, default=None, required=False)
    parser.add_argument("--h", help="Height of image crop", type=int, default=None, required=False)
    parser.add_argument("--i", help="Index of image to inspect", type=int, default=0, required=False)
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        exit("Input folder '{}' does not exist!".format(args.folder))

    crop_size = None
    if args.w != None and args.h != None:
        crop_size = (args.w, args.h)

    sceneflow = sceneflow_loader(root=args.folder, split=args.split, crop_size=crop_size)
    img_l, img_r, img_d = sceneflow.__getitem__(args.i)

    img_d = (img_d * 256).astype('uint16')
    img_d = Image.fromarray(img_d)
    print("[INFO]: example disparity saved.")
    img_d.save("example_disparity.png")


if __name__ == "__main__":
    main()