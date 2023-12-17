import torch
import numpy as np
import torchvision
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset
import pickle
from sentence_transformers import SentenceTransformer, util
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from torchvision.utils import make_grid
from sklearn.decomposition import PCA

class PuzzleImageDataset(Dataset):
    '''
        Assume that puzzle images always consist of 6*6 parts numbered in the predefined convention.
        Returns each puzzle components as a representation comes out of a pretrained model,
        in a combined manner.
        So for a given index(puzzle) output will be (36, hidden_dim)
    '''

    def __init__(self, img_dir, data_file=None, label_file_name=None, grid_size=6):

        self.img_labels = {}

        if(label_file_name):
            with open(os.path.join(os.path.join(img_dir, label_file_name)), 'r') as f:
                content = f.readlines()
            for i, val in enumerate(content):
                self.img_labels[i] = int(val)

        self.img_dir = img_dir
        self.grid_size = grid_size

        #Load CLIP model
        weights = ResNet50_Weights.DEFAULT
        self.preprocess_func = weights.transforms(antialias=True)
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.eval()

        if(data_file):
            with open(data_file, 'rb') as f:
                self.data_file = pickle.load(f)
        else:
            self.data_file = None


    def __len__(self):
        return len([name for name in os.listdir(self.img_dir) if name.endswith('.jpg') and "_0." in name])

    def __getitem__(self, idx):
        
        if not (self.data_file):
            return -1, -1
        
        vals = self.data_file[idx]

        if(vals['label'] is None):
            return vals['img_vecs'], vals['img_grid_vec'], vals['similarity_features'], -1
            
        return vals['img_vecs'], vals['img_grid'], vals['img_grid_vec'], vals['similarity_features'], vals['label']
    
    
    def preprocess(self, device, data_out_name, side_range=20):
        print("Preprocessing of data started!")
        output_dict = {}
        if(device):
            self.model.to(device)

        # pca = PCA(n_components=36)

        for idx in range(len([name for name in os.listdir(self.img_dir) if name.endswith('.jpg') and "_0." in name])):
            
            imgs = []
            img_tops = []
            img_bottoms = []
            img_lefts = []
            img_rights = []

            for i in range(self.grid_size**2):
                img_path = os.path.join(self.img_dir, f"{idx}_{i}.jpg")
                image  = Image.open(img_path)

                # Getting the puzzle piece pixels
                
                image = np.array(image)
                imgs.append(image)

                # Getting the puzzle piece edge segments
                h, w, c = image.shape
                image_top = image[:side_range, :, :]
                image_bottom = image[h-side_range:, :, :]
                image_left = image[:, :side_range, :]
                image_right = image[:, w-side_range:, :]

                img_tops.append(image_top)
                img_bottoms.append(image_bottom)
                img_lefts.append(image_left)
                img_rights.append(image_right)

            # Getting the puzzle piece combined Image
            temp_arr = [torch.from_numpy(np.array(i)).permute((2, 0, 1)) for i in imgs]
            img_grid_vec = make_grid(temp_arr, nrow=6, padding=0)
            img_grid = img_grid_vec.permute(((1, 2, 0))).numpy()
            img_grid = Image.fromarray(img_grid.astype('uint8'), 'RGB')

            preprocessed_imgs = self.preprocess_func(torch.from_numpy(np.array(imgs)).permute((0, 3, 1, 2)))
            img_vecs = self.model(preprocessed_imgs.to(device)).squeeze()

            preprocessed_imgs = self.preprocess_func(torch.from_numpy(np.array(img_lefts)).permute((0, 3, 1, 2)))
            img_left_vecs = self.model(preprocessed_imgs.to(device)).squeeze()

            preprocessed_imgs = self.preprocess_func(torch.from_numpy(np.array(img_rights)).permute((0, 3, 1, 2)))
            img_right_vecs = self.model(preprocessed_imgs.to(device)).squeeze()

            preprocessed_imgs = self.preprocess_func(torch.from_numpy(np.array(img_tops)).permute((0, 3, 1, 2)))
            img_top_vecs = self.model(preprocessed_imgs.to(device)).squeeze()

            preprocessed_imgs = self.preprocess_func(torch.from_numpy(np.array(img_bottoms)).permute((0, 3, 1, 2)))
            img_bottom_vecs = self.model(preprocessed_imgs.to(device)).squeeze()

            piece_similarity_scores = util.cos_sim(img_vecs, img_vecs)
            left_right_similarity_scores = util.cos_sim(img_left_vecs, img_right_vecs)
            top_bottom_similarity_scores = util.cos_sim(img_top_vecs, img_bottom_vecs)
            left_left_similarity_scores = util.cos_sim(img_left_vecs, img_left_vecs)
            right_right_similarity_scores = util.cos_sim(img_right_vecs, img_right_vecs)
            top_top_similarity_scores = util.cos_sim(img_top_vecs, img_top_vecs)
            bottom_bottom_similarity_scores = util.cos_sim(img_bottom_vecs, img_bottom_vecs)

            # reduced_features = pca.fit_transform(img_vecs.detach().cpu().numpy())
   
            similarity_features = torch.stack(( piece_similarity_scores,
                                                # torch.from_numpy(reduced_features).to(device),
                                                left_right_similarity_scores, 
                                                top_bottom_similarity_scores,
                                                # left_left_similarity_scores,
                                                # right_right_similarity_scores,
                                                # top_top_similarity_scores, 
                                                # bottom_bottom_similarity_scores
                                                ))

            if(idx%10==9):
                print(f"{idx+1} completed.")

            label = self.img_labels.get(idx, -1)
            output_dict[idx] = {"img_vecs": img_vecs.detach().cpu().numpy(), # (36, 1024)
                                "img_grid_vec": img_grid_vec.detach().cpu().numpy(), # (1, 1024)
                                "img_grid": img_grid, # (3, 336, 336)
                                "similarity_features": similarity_features.detach().cpu().numpy(), # (36, 36)
                                "label": label} # 1
            
        print("processing completed!")
        with open(f"{data_out_name}.pkl", "wb") as f:
            pickle.dump(output_dict, f)

        self.data_file = output_dict

        return

class PermutationPredictor(nn.Module):


    def __init__(self, num_heads=1, dropout=0):
        super(PermutationPredictor, self).__init__()

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 36, kernel_size=1, stride=1)
        self.conv_drop = nn.Dropout2d(dropout)

        self.map_extender = nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2)
        self.map_extender1 = nn.Conv2d(16, 64, kernel_size=5, stride=1, padding=2)
        self.extender_drop = nn.Dropout2d(dropout)

        self.attention_lyr1 = nn.MultiheadAttention(36*8, num_heads, batch_first=True, dropout=dropout)
        self.normalize_lyr1 = nn.LayerNorm(36*8)
        self.ff_1 = nn.Linear(36*8, 36*8)
        self.ff_normalize_lyr1 = nn.LayerNorm(36*8)
        self.attention_drop1 = nn.Dropout(dropout)

        self.attention_lyr2 = nn.MultiheadAttention(36*8, num_heads, batch_first=True, dropout=dropout)
        self.normalize_lyr2 = nn.LayerNorm(36*8)
        self.ff_2 = nn.Linear(36*8, 36*8)
        self.ff_normalize_lyr2 = nn.LayerNorm(36*8)
        self.attention_drop2 = nn.Dropout(dropout)

        self.flatten_lyr = nn.Flatten()
        self.linear1_lyr = nn.Linear(36*8*36*8, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2_lyr = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout)

        self.output_lyr = nn.Linear(128, 50)

    def forward(self, img_vecs, similarity_features):
        '''
            input include the similarity maps on pixel values.

            img_vecs: (-1, 36, 2048)
            similarity_features: (-1, 3, 36, 36)

        '''
        img_vecs = img_vecs.permute((0, 2, 1)) # b, 2048, 36
        img_vecs = img_vecs.unsqueeze(-1) # b, 2048, 36, 1 

        img_vecs = self.conv1(img_vecs)
        # img_vecs = nn.functional.tanh(img_vecs)
        img_vecs = self.conv2(img_vecs)
        # img_vecs = nn.functional.tanh(img_vecs)
        img_vecs = self.conv3(img_vecs)
        img_vecs = nn.functional.leaky_relu(img_vecs)
        img_vecs =  self.conv_drop(img_vecs)

        img_vecs = img_vecs.permute((0, 3, 1, 2))
        feature_maps = torch.concat((img_vecs, similarity_features), dim=1)

        feature_maps = self.map_extender(feature_maps)
        feature_maps = self.map_extender1(feature_maps)
        feature_maps = nn.functional.leaky_relu(feature_maps)
        feature_maps = self.extender_drop(feature_maps)
        feature_maps = feature_maps.permute((0, 2, 3, 1))
        feature_maps = feature_maps.reshape((-1, 36*8, 36*8))
 
        weighted_atten, _ = self.attention_lyr1(feature_maps, feature_maps, feature_maps)
        feature_maps = self.normalize_lyr1(feature_maps + weighted_atten)
        temp = self.ff_1(feature_maps)
        feature_maps = self.ff_normalize_lyr1(feature_maps+temp)
        feature_maps = self.attention_drop1(feature_maps)

        feature_maps = feature_maps.permute((0, 2, 1))
        weighted_atten, _ = self.attention_lyr2(feature_maps, feature_maps, feature_maps)
        feature_maps = self.normalize_lyr2(feature_maps + weighted_atten)
        temp = self.ff_2(feature_maps)
        feature_maps = self.ff_normalize_lyr2(feature_maps+temp)
        feature_maps = self.attention_drop2(feature_maps)

        feature_maps = self.flatten_lyr(feature_maps)

        feature_maps = self.linear1_lyr(feature_maps)
        feature_maps = nn.functional.leaky_relu(feature_maps)
        feature_maps = self.dropout1(feature_maps)

        feature_maps = self.linear2_lyr(feature_maps)
        feature_maps = nn.functional.leaky_relu(feature_maps)
        feature_maps = self.dropout2(feature_maps)

        feature_maps = self.output_lyr(feature_maps)
        return feature_maps
