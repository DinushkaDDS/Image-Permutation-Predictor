import os
import matplotlib.pyplot as plt
import os
from torchvision.io import read_image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.utils import make_grid

def predict_img(model, preprocess_fn, img_dir, img_idx, device, grid_size=6):

    imgs = []
    for i in range(grid_size**2):
        img_path = os.path.join(img_dir, f"{img_idx}_{i}.jpg")
        image = read_image(img_path)
        imgs.append(image)
    
    img_grid = make_grid(imgs, nrow=grid_size, padding=0)
    print(img_grid.unsqueeze(0).shape)
    imgs_gpu = preprocess_fn(img_grid.unsqueeze(0).to(device))

    output = model(imgs_gpu) 
    pred_lbl = torch.argmax(nn.functional.softmax(output, dim=-1), dim=-1).tolist()
    image_show(img_grid, f"Predicted Label: {pred_lbl[0]}")

def image_show(img, label=None):
    '''
        Function to display a single image.
    '''
    figure = plt.figure()

    if(label):
        figure.suptitle(f"{label}", fontsize=12)
    
    img = torch.from_numpy(img)
    image = img.permute((1, 2, 0))
    plt.axis("off")
    plt.imshow(image.squeeze(), cmap="gray")
    plt.show()