'''
This is the combined pipeline for training and inference.
'''
import math

import torch
import torch.nn as nn

import params
from core.dcnet import DCnet
from core.smnet import SMnet
from utils.loss import VDT_loss

import sys
sys.path.append("../utils")
from utils.general import get_time_stamp


class VDT(nn.Module):
    def __init__(self):
        super(VDT, self).__init__()
        
        self.dcnet = DCnet()
        self.smnet = SMnet()

        self.optimizer = torch.optim.Adam(self.parameters(), lr = params.LEARNING_RATE)

        #TODO::add parameters to set weight for each loss term
        self.loss_func = VDT_loss()

    def name(self): return "VDT"


    def load_checkpoint(self, file_path):
        checkpoint = torch.load(file_path)

        self.smnet.load_state_dict(checkpoint["smnet"])
        self.dcnet.load_state_dict(checkpoint["dcnet"])

        print("Loaded checkpoint from file '{}'".format(file_path))


    def load_params(self, dcnet_params, smnet_params, rpnet_params):
        self.dcnet.load_state_dict(dcnet_params)
        self.smnet.load_state_dict(smnet_params)


    def save_checkpoint(self, file_path=None):
        if file_path == None:
            prefix = "VDT"
            file_path = "./training/{0}_{1}.pth.tar".format(prefix, get_time_stamp())
        
        checkpoint = {}
        checkpoint["smnet"] = self.smnet.state_dict()
        checkpoint["dcnet"] = self.dcnet.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        
        torch.save(checkpoint, file_path)
        print("Saved checkpoint {}...\n".format(file_path))


    def train_networks(self, l, r, t = None, T = None):
        '''
        this is the training function of the pipeline.
        l - input left image
        r - input right iamge
        t - input disparity ground truth
        T - input depth ground truth
        '''

        # forward
        d = self.smnet(l, r)
        
        d[d == 0] = 1 / 1e6
        d = torch.reciprocal(d) # disparity to depth conversion

        D = self.dcnet(l, d)
        loss = self.loss_func(l, r, d, D, t, T)

        # backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()


    def forward(self, l, r):
        '''run VDT pipeline'''
        
        d = self.smnet(l, r)
        
        # disparity to depth conversion
        d[d == 0] = 1 / 1e6
        d = torch.reciprocal(d)

        y = self.dcnet(l, d)
        return y


def main():
    model = VDT()
    
    l = torch.rand(1, 3, 128, 128)
    r = torch.rand(1, 3, 128, 128)
    print(l.shape)

    D = model(l, r)
    print(D)


if __name__ == "__main__":
    main()
