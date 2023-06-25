import torchvision.transforms as transforms


__imagenet_stats = { 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225] }


def normalize(normalize=__imagenet_stats):
    '''normalize the input iamge with the imagenet statistics'''

    list_of_transform = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(list_of_transform)