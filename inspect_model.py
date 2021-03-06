import torch
from prettytable import PrettyTable

from core.dcnet import DCnet


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    
    print(table)
    print(f"Total Trainable Params: {total_params}")

    return total_params


def main():
    model = DCnet()
    print("[INFO]: Model created successfully!")
    count_parameters(model)


if __name__ == "__main__":
    main()