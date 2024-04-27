import os

data_path = "/home/pinxuan/Documents/gaussian_surface_reconstruction/data/blendedMVS"

# all_scan = [i for i in os.listdir(data_path)]
all_scan = ['gold']

for i in all_scan:
    print(f"running: {i}")
    os.system(f"python test.py --img_path {data_path}/{i}/image --output_path {data_path}/{i}/depth --task depth")
    os.system(f"python test.py --img_path {data_path}/{i}/image --output_path {data_path}/{i}/normal --task normal")
    # os.system(f"CUDA_VISIBLE_DEVICES=1 python /home/pinxuan/Documents/gaussian_surface_reconstruction/train.py -s {dtu_path}/{i} --iterations 15000")
