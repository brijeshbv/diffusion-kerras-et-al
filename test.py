from k_diffusion import models
import torch

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