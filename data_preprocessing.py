import os
import torch
import torchvision
from torchvision.io import read_image

def process_directory(img_dir, label_file_name, grid_size=6):
    '''
        Will return multiple sized vectors for each of the feature map
    '''

    img_labels = {}
    with open(os.path.join(os.path.join(img_dir, label_file_name)), 'r') as f:
        content = f.readlines()

    for i, val in enumerate(content):
        img_labels[i] = int(val)

    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).backbone
    preprocess_fn = weights.transforms()

    for idx in range(len(img_labels)):

        puzzle_representation = []
        for i in range(grid_size**2):
            img_path = os.path.join(img_dir, f"{idx}_{i}.jpg")
            image = read_image(img_path)


