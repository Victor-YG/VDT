import os
import cv2
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def load_image(file_path):
    img = Image.open(file_path)
    img_arr = np.array(img)
    return np.divide(img, 255.0)


def load_depth(file_path):
    return np.array(Image.open(file_path), dtype=float)


class KITTI_Frame():
    '''this object contains the kitti data for a single time stamp'''

    def __init__(self, l_c, r_c=None, l_d=None, r_d=None, l_t=None, r_t=None):
        self.l_c = l_c
        self.r_c = r_c
        self.l_d = l_d
        self.r_d = r_d
        self.l_t = l_t
        self.r_t = r_t


    def load_frame(self):
        frame = {}
        frame["l_c"] = load_image(self.l_c) if self.l_c != None else None
        frame["r_c"] = load_image(self.r_c) if self.r_c != None else None
        frame["l_d"] = load_depth(self.l_d) if self.l_d != None else None
        frame["r_d"] = load_depth(self.r_d) if self.r_d != None else None
        frame["l_t"] = load_depth(self.l_t) if self.l_t != None else None
        frame["r_t"] = load_depth(self.r_t) if self.r_t != None else None
        return frame


class KITTI_Drive():
    '''this object contains the kitti data for a particular drive'''

    def __init__(self, folder):
        print("Loading data from '{}'...".format(folder))
        self.data = []
        
        if not os.path.exists(folder):
            print("folder '{}' does not exist.".format(folder))
            return
        
        left_img_folder = os.path.join(folder, r"image_02\data")
        for l_c in os.listdir(left_img_folder):
            self.data.append(self.__match__(os.path.join(left_img_folder, l_c)))


    def __match__(self,  l_c):
        if not os.path.exists(l_c):
            raise FileNotFoundError("left rgb image '{}' is not found!".format(l_c))
        frame = KITTI_Frame(l_c)
        
        r_c = l_c.replace(r"image_02", r"image_03")
        if os.path.exists(r_c):
            frame.r_c = r_c
        
        l_d = l_c.replace(r"image_02\data", r"proj_depth\velodyne_raw\image_02")
        if os.path.exists(l_d):
            frame.l_d   = l_d
        
        r_d = l_c.replace(r"image_02\data", r"proj_depth\velodyne_raw\image_03")
        if os.path.exists(r_d):
            frame.r_d   = r_d
        
        l_t = l_c.replace(r"image_02\data", r"proj_depth\groundtruth\image_02")
        if os.path.exists(l_t):
            frame.l_t   = l_t

        r_t = l_c.replace(r"image_02\data", r"proj_depth\groundtruth\image_03")
        if os.path.exists(r_t):
            frame.r_t   = r_t
        
        return frame


class KITTI_Test():
    '''this object contains the kitti test data for a particular application (completion or prediction)'''

    def __init__(self, folder):
        self.data = []
        
        if not os.path.exists(folder):
            print("folder '{}' does not exist.".format(folder))
            return
        
        left_img_folder = os.path.join(folder, r"image")
        for l_c in os.listdir(left_img_folder):
            self.data.append(self.__match__(os.path.join(left_img_folder, l_c)))


    def __match__(self, l_c):
        frame = KITTI_Frame()

        if os.path.exists(l_c):
            frame.l_c = l_c
        
        l_d = l_c.replace(r"image", r"velodyne_raw")
        if os.path.exists(l_d):
            frame.l_d   = l_d
        
        return frame


class KITTI(Dataset):
    '''this object manage the KITTI dataset'''

    # potential split: "train", "validation", "test_completion", "test_prediction"
    # potential mode:  "monocular", "monodepth", "stereo", "stereodepth"
    def __init__(self, folder, transform, mode="monocular", split="train"):
        print("Loading KITTI dataset for split '{}'...".format(split))

        if not os.path.exists(folder):
            print("folder '{}' does not exist.".format(folder))
            return
        
        self.kitti_folder    = folder
        self.transform       = transform
        self.mode            = mode
        self.split           = split
        self.data            = []
        
        self.__load_calib__(split)
        self.__load__(split)

        print("Found {} sets of images in total for split '{}'...".format(len(self.data), split))


    def __load_calib__(self, split):
        pass


    def __load__(self, split):
        # load training data
        if   split == "train":
            folder = os.path.join(self.kitti_folder, "train")
        elif split == "validation":
            folder = os.path.join(self.kitti_folder, "val")
        elif split == "test_completion":
            folder = os.path.join(self.kitti_folder, "test_depth_completion_anonymous")
        elif split == "test_prediction":
            folder = os.path.join(self.kitti_folder, "test_depth_prediction_anonymous")
        else:
            raise ValueError("Unexpected split tag used '{}'!".format(split))

        if not os.path.exists(folder):
            print("folder '{}' does not exist.".format(folder))
            return

        if split == "train" or split == "validation":
            for drive_folder in os.listdir(folder):
                drive = KITTI_Drive(os.path.join(folder, drive_folder))
                self.__prep__(drive.data)
        else:
            test = KITTI_Test(folder)
            self.__prep__(test.data)


    def __prep__(self, data):
        for frame in data:
            if self.mode == "monocular":
                if frame.l_c != None and frame.l_t != None:
                    data_1 = {}
                    data_1["c"] = frame.l_c
                    data_1["t"] = frame.l_t
                    self.data.append(data_1)
                if frame.r_c != None and frame.r_t != None:
                    data_2 = {}
                    data_2["c"] = frame.r_c
                    data_2["t"] = frame.r_t
                    self.data.append(data_2)

            if self.mode == "monodepth":
                if frame.l_c != None and frame.l_d != None and frame.l_t != None:
                    data_1 = {}
                    data_1["c"] = frame.l_c
                    data_1["d"] = frame.l_d
                    data_1["t"] = frame.l_t
                    self.data.append(data_1)
                if frame.r_c != None and frame.r_d != None and frame.r_t != None:
                    data_2 = {}
                    data_2["c"] = frame.r_c
                    data_2["d"] = frame.r_d
                    data_2["t"] = frame.r_t
                    self.data.append(data_2)

            if self.mode == "stereo":
                if frame.l_c != None and frame.r_c != None and frame.l_d != None and frame.l_t != None:
                    data_1 = {}
                    data_1["l_c"] = frame.l_c
                    data_1["r_c"] = frame.r_c
                    data_1["d"]   = frame.l_d
                    data_1["t"]   = frame.l_t
                    self.data.append(data_1)

            if self.mode == "stereodepth":
                if frame.l_c != None and frame.r_c != None and frame.l_d != None and frame.r_d != None and frame.l_t != None and frame.r_t != None:
                    data_1 = {}
                    data_1["l_c"] = frame.l_c
                    data_1["r_c"] = frame.r_c
                    data_1["l_d"] = frame.l_d
                    data_1["l_t"] = frame.l_t
                    data_1["r_d"] = frame.r_d
                    data_1["r_t"] = frame.r_t
                    self.data.append(data_1)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder containing KITTI dataset", required=True)
    parser.add_argument("--split", help="What type of data (split) to load", default="train", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        exit("Input folder '{}' does not exist!".format(args.folder))

    kitti = KITTI(args.folder, args.split, mode="monodepth")


if __name__ == "__main__":
    main()