import os
import cv2
import time
import torch
import argparse
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from core.models.dcnet import DCnet

# from utils.io.ply_utils import depth_map_to_ply


# check hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-weights", help="Weight for network.", default=r".\weights\dcnet.pth.tar", required=False)
    parser.add_argument("-rgb", help="Input RGB image.", default=r".\test\0000000005_c.png", required=False)
    parser.add_argument("-depth", help="Input depth image.", default=r".\test\0000000005_d.png", required=False)
    parser.add_argument("-output", help="Output folder.", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.rgb):
        exit("Input rgb image '{}' not found!".format(args.rgb))
    
    if not os.path.exists(args.depth):
        exit("Input depth image '{}' not found!".format(args.depth))
    
    ##############
    # load input #
    ##############
    img_c_original = Image.open(args.rgb)
    img_d_original = Image.open(args.depth)
    w_input = img_c_original.width
    h_input = img_c_original.height
    print("Original image size (w x h) = ({}, {})".format(w_input, h_input))

    # TODO::find way to avoid croping the image
    img_c = img_c_original.crop([0, 0, 1024, 256])
    img_d = img_d_original.crop([0, 0, 1024, 256])

    # convert imgs to numpy float array
    arr_c = np.array(img_c, dtype=np.float32)
    arr_d = np.array(img_d, dtype=np.float32)

    # convert to tensor
    tensor_c = torch.tensor(arr_c)
    tensor_d = torch.tensor(arr_d)

    # scale and transform value to between 0 and 1
    tensor_c.div_(255.0)
    tensor_d.div_(256.0)

    # adjust dimension and concat
    print("Input image size = {}".format(tensor_c.shape))
    tensor_c = torch.reshape(tensor_c, [-1, tensor_c.shape[0], tensor_c.shape[1], tensor_c.shape[2]])
    tensor_c = torch.einsum("nhwc->nchw", tensor_c)
    tensor_d = torch.reshape(tensor_d, [-1, 1, tensor_d.shape[0], tensor_d.shape[1]])    
    tensor_c = tensor_c.to(DEVICE)
    tensor_d = tensor_d.to(DEVICE)

    ################
    # create model #
    ################
    model = DCnet().to(DEVICE)
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint["model"])
    print("Loaded checkpoint from file '{}'".format(args.weights))

    #############
    # run model #
    #############

    with torch.no_grad():
        start_time = time.monotonic()
        tensor_p = model(tensor_c, tensor_d)
        end_time = time.monotonic()

        arr_pred = tensor_p.cpu().numpy()
        arr_pred = np.einsum("nchw->hw", arr_pred)
        img_p = Image.fromarray(arr_pred)

    print("Time spent is {:.3f} secs.".format(end_time - start_time))

    ###############
    # save output #
    ###############
    img_final = Image.new("RGB", (w_input, h_input * 3), "black")
    img_final.paste(img_c, (0, 0))
    img_final.paste(img_d, (0, h_input))
    img_final.paste(img_p, (0, h_input * 2))
    img_final.show()

    if args.output:
        img_final_path = os.path.basename(args.rgb) + "_p.png"
        img_final.save(os.path.join(args.output, img_final_path))


if __name__ == "__main__":
    main()