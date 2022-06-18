import os
import time
import argparse

import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from VDT import VDT


# check hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-weights", help="Weight for network.", default=r".\weights\VDT.pth.tar", required=False)
    parser.add_argument("-img_l", help="Input RGB image.", default=r".\test\0000000005_l.png", required=False)
    parser.add_argument("-img_r", help="Input depth image.", default=r".\test\0000000005_r.png", required=False)
    parser.add_argument("-img_T", help="Input detph ground truth.", default=r".\test\0000000005_l_T.png", required=False)
    parser.add_argument("-output", help="Output folder.", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.img_l):
        exit("Input left image '{}' not found!".format(args.img_l))
    
    if not os.path.exists(args.img_r):
        exit("Input right image '{}' not found!".format(args.img_r))
    
    ##############
    # load input #
    ##############
    img_l = Image.open(args.img_l)
    img_r = Image.open(args.img_r)
    img_T = Image.open(args.img_T)

    w_input = img_l.width
    h_input = img_l.height
    print("Original image size (w x h) = ({}, {})".format(w_input, h_input))
    
    # convert imgs to numpy float array
    arr_l = np.array(img_l, dtype=np.float32)
    arr_r = np.array(img_r, dtype=np.float32)

    # convert to tensor
    tensor_l = torch.tensor(arr_l)
    tensor_r = torch.tensor(arr_r)

    # scale and transform value to between 0 and 1
    tensor_l.div_(255.0)
    tensor_r.div_(255.0)

    # adjust dimension and concat
    print("Input image size = {}".format(tensor_l.shape))
    tensor_l = torch.reshape(tensor_l, [-1, tensor_l.shape[0], tensor_l.shape[1], tensor_l.shape[2]])
    tensor_r = torch.reshape(tensor_r, [-1, tensor_r.shape[0], tensor_r.shape[1], tensor_r.shape[2]])
    tensor_l = torch.einsum("nhwc->nchw", tensor_l)
    tensor_r = torch.einsum("nhwc->nchw", tensor_r)
    tensor_l = tensor_l.to(DEVICE)
    tensor_r = tensor_r.to(DEVICE)

    ################
    # create model #
    ################
    model = VDT().to(DEVICE)
    model.load_checkpoint(args.weights)

    #############
    # run model #
    #############
    with torch.no_grad():
        start_time = time.monotonic()
        tensor_p = model(tensor_l, tensor_r)
        end_time = time.monotonic()

        arr_pred = tensor_p.cpu().numpy()
        arr_pred = np.einsum("nchw->hw", arr_pred)
        img_p = Image.fromarray(arr_pred)

    print("Time spent is {:.3f} secs.".format(end_time - start_time))

    ###############
    # save output #
    ###############
    img_final = Image.new("RGB", (w_input, h_input * 4), "black")
    img_final.paste(img_l, (0, 0))
    img_final.paste(img_r, (0, h_input))
    img_final.paste(img_p, (0, h_input * 2))
    img_final.paste(img_T, (0, h_input * 3))
    img_final.show()

    if args.output:
        img_final_path = os.path.basename(args.rgb) + "_p.png"
        img_final.save(os.path.join(args.output, img_final_path))


if __name__ == "__main__":
    main()
