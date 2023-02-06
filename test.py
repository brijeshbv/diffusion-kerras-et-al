from k_diffusion import models
import torch
from copy import deepcopy
import json
import math
from pathlib import Path
#import accelerate
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import transforms, utils as tv_utils
from tqdm import trange
import datasets # HF version
from matplotlib import pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from k_diffusion import evaluation, gns, layers, models, sampling, utils

class CelebADataset(Dataset):
    def __init__(self, img_size=128):
        self.dataset = load_dataset('huggan/CelebA-faces', split='train')
        self.preprocess = transforms.Compose([transforms.ToTensor(),transforms.Resize(img_size), transforms.CenterCrop(img_size)])
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x = self.dataset[idx]
        return self.preprocess(x['image']) * 2 - 1 # Images scaled to (-1, 1)

img_size = 64
batch_size=16
dataset = CelebADataset(img_size)
dl = DataLoader(dataset, batch_size=batch_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

inner_model_edm = models.ImageDenoiserModelV1(
    3, # input channels
    256, # mapping out
    [2, 2, 4], # depths
    [64, 128, 256], # channels
    [False, True, True] # self attention
).to(device)
#print(inner_model_edm)


inner_model_vpsong = models.SongUNet(
    64,
    3,
    3
).to(device)
print(inner_model_vpsong)

batch = next(iter(dl))
print('Batch shape:', batch.shape)
utils.to_pil_image(batch[3]) # View the first image

# Duplicate the first image 12 times for demo purposes:
input_images = batch[3].unsqueeze(0).repeat(12, 1, 1, 1)

# Add noise (linearly)
reals = input_images.to(device)
noise = torch.randn_like(reals)
sigma = torch.linspace(0, 3, 12).to(device) # Gradually increasing noise
noised_input = reals + noise * utils.append_dims(sigma, reals.ndim)

# View the result
utils.to_pil_image(tv_utils.make_grid(noised_input))

sigma_mean, sigma_std = -1.2, 1.2
reals = input_images.to(device)
noise = torch.randn_like(reals)
sigma = torch.distributions.LogNormal(sigma_mean, sigma_std).sample([reals.shape[0]]).to(device)
noised_input = reals + noise * utils.append_dims(sigma, reals.ndim)
utils.to_pil_image(tv_utils.make_grid(noised_input))


inner_model_edm_output = inner_model_edm(noised_input, sigma)
print(inner_model_edm_output.shape)

inner_model_vpsong_output = inner_model_vpsong(noised_input, sigma)
print(inner_model_vpsong_output.shape)