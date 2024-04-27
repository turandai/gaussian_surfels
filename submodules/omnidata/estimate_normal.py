import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys
from tqdm import tqdm

from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = './pretrained_models/'

trans_topil = transforms.ToPILImage()

# print(args.output_path)
if args.task == 'NONE':
    args.task = 'normal'
if args.output_path is None:
    args.output_path = f"{args.img_path}/../{args.task}"
os.makedirs(args.output_path, exist_ok=True)
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if args.task == 'normal':
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

elif args.task == 'depth':

    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)


@torch.no_grad()
def save_outputs(img_path, output_file_name):
    save_path = os.path.join(args.output_path, f'{output_file_name}_{args.task}.png')
    npy_path = os.path.join(args.output_path, f"{output_file_name}.npy")

    # print(f'Reading input {img_path} ...')
    img = Image.open(img_path)

    WIDTH, HEIGHT = img.size
    # print("Load image shape ", HEIGHT, WIDTH)
    # exit()

    INPUT_WIDTH = WIDTH // 32 * 32
    INPUT_HEIGHT = HEIGHT // 32 * 32
    # INPUT_WIDTH = 1280 // 2
    # INPUT_HEIGHT = 960 //2
    # print("Processing shape ", INPUT_HEIGHT, INPUT_WIDTH)

    if args.task == 'depth':
        trans_totensor = transforms.Compose([transforms.Resize([INPUT_HEIGHT, INPUT_WIDTH], interpolation=PIL.Image.BILINEAR),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=0.5, std=0.5)])
    elif args.task == 'normal':
        trans_totensor = transforms.Compose([transforms.Resize([INPUT_HEIGHT, INPUT_WIDTH], interpolation=PIL.Image.BILINEAR),
                                            get_transform('rgb', image_size=None)])
    
    img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

    # rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
    # img.save(rgb_path)

    if img_tensor.shape[1] == 1:
        img_tensor = img_tensor.repeat_interleave(3,1)

    output = model(img_tensor).clamp(min=0, max=1)
    # print(output.shape)

    if args.task == 'depth':

        output = F.interpolate(output.unsqueeze(0), (HEIGHT, WIDTH), mode='bicubic').squeeze(0)
        output = output.clamp(0,1)
        np.save(os.path.join(args.output_path, f'{output_file_name}_depth.npy'), output.to('cpu').numpy())

        # # output = 1 - output
        # plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')

        # output = output.detach().cpu().squeeze().numpy()
        # output = (output - output.min()) / (output.max() - output.min())
        # np.save(npy_path, output)

    else:
        output = F.interpolate(output, (HEIGHT, WIDTH), mode='bilinear')
        # print(output.shape)
        # exit()
        output = torch.nn.functional.normalize(output * 2 - 1, dim=1)
        np.save(os.path.join(args.output_path, f'{output_file_name}_normal.npy'), output.to('cpu').numpy())
        # trans_topil(output[0]).save(save_path)
        # output = output[0].permute(1,2,0).detach().cpu().numpy()
        # output = output * 2. - 1.
        # output = output / (np.linalg.norm(output, axis=-1)[..., None] + 1e-15)
        # np.save(npy_path, output)
        # raise NotImplementedError
    # print(f'Writing output {save_path} ...')

img_path = Path(args.img_path)
# if img_path.is_file():
#     save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
if img_path.is_dir():
    images = glob.glob(args.img_path+'/*.jpg') + glob.glob(args.img_path+'/*.png') + glob.glob(args.img_path+'/*.JPG')
    for f in tqdm(images, desc=f"Estimating {args.task}"):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
        