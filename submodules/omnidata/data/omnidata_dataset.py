from   collections import namedtuple, Counter, defaultdict
from   dataclasses import dataclass, field
from   joblib import Parallel, delayed
import logging
import multiprocessing as mp
import os
import pickle
import json
from   PIL import Image, ImageFile
import pandas as pd
import random
import re
from   time import perf_counter 
from   tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F
from   torchvision import transforms
import torchvision.transforms.functional as TF
from   typing import Optional, List, Callable, Union, Dict, Any
import warnings

from .taskonomy_dataset import parse_filename, LabelFile, View
from .splits import taskonomy_flat_split_to_buildings, replica_flat_split_to_buildings, \
    gso_flat_split_to_buildings, hypersim_flat_split_to_buildings, blendedMVS_flat_split_to_buildings
from .transforms import default_loader, get_transform, LocalContrastNormalization
from .task_configs import task_parameters, SINGLE_IMAGE_TASKS
from .refocus_augmentation import RefocusImageAugmentation

ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO Test this

MAX_VIEWS = 45

RGB_MEAN = torch.Tensor([0.55312, 0.52514, 0.49313]).reshape(3,1,1)
RGB_STD =  torch.Tensor([0.20555, 0.21775, 0.24044]).reshape(3,1,1)

REPLICA_BUILDINGS = [
    'frl_apartment_5', 'office_2', 'room_2', 'office_4', 'frl_apartment_0', 'frl_apartment_4',
    'office_1', 'frl_apartment_3', 'office_0', 'apartment_2', 'room_0', 'apartment_1', 
    'frl_apartment_1', 'office_3', 'frl_apartment_2', 'apartment_0', 'hotel_0', 'room_1']

def filter_taskonomy():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset_stats', 'taskonomy_sorted.csv.gz'), index_col=0)
    thresholds = [10000, 50000, 100000]
    filtered_bpv = {}
    for thresh in thresholds:
        count = 0
        d = defaultdict(lambda: defaultdict(list))
        for index, row in data.iterrows():
            if count > thresh: break
            d[row['building']][int(row['point'])].append(int(row['view']))
            count += 1
        filtered_bpv[thresh] = d
    return filtered_bpv

def filter_blendedMVS():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset_stats', 'blendedMVS_sorted.csv.gz'), index_col=0)
    thresholds = [1000, 10000, 20000]
    filtered_bpv = {}
    for thresh in thresholds:
        count = 0
        d = defaultdict(list)
        for index, row in data.iterrows():
            if count > thresh: break
            d[row['building']].append(int(row['point']))
            count += 1
        filtered_bpv[thresh] = d
    return filtered_bpv

blendedMVS_bad_scenes = ['5bf21799d43923194842c001',
                        '000000000000000000000001',
                        '000000000000000000000000',
                        '5b21e18c58e2823a67a10dd8',
                        '5ab8b8e029f5351f7f2ccf59',
                        '00000000000000000000000c',
                        '00000000000000000000000b',
                        '58cf4771d0f5fb221defe6da',
                        '5c2b3ed5e611832e8aed46bf',
                        '584a7333fe3cb463906c9fe6',
                        '585ee0632a57cc11d4933608',
                        '5983012d1bd4b175e70c985a',
                        '58a0dd1a3d0b4542479a28f3',
                        '58a44463156b87103d3ed45e']

                    
class OmnidataDataset(data.Dataset):
    '''
        Loads data for the Taskonomy dataset.
        This expects that the data is structured
        
            /path/to/data/
                rgb/
                    modelk/
                        point_i_view_j.png
                        ...                        
                depth_euclidean/
                ... (other tasks)
                
        If one would like to use pretrained representations, then they can be added into the directory as:
            /path/to/data/
                rgb_encoding/
                    modelk/
                        point_i_view_j.npy
                ...
        
        Basically, any other folder name will work as long as it is named the same way.
    '''
    @dataclass
    class Options():
        '''
            data_path: Path to data
            tasks: Which tasks to load. Any subfolder will work as long as data is named accordingly
            buildings: Which models to include. See `splits.taskonomy` (can also be a string, e.g. 'fullplus-val')
            transform: one transform per task.
            
            Note: This assumes that all images are present in all (used) subfolders
        '''
        taskonomy_data_path: str = '/datasets/taskonomy'
        replica_data_path: str = '/scratch/ainaz/replica-taskonomized'
        gso_data_path: str = '/scratch/ainaz/replica-google-objects'
        hypersim_data_path: str = '/scratch/ainaz/hypersim-dataset2/evermotion/scenes'
        blendedMVS_data_path: str = '/scratch/ainaz/BlendedMVS/mvs_low_res_taskonomized'
        split: str = 'train'
        taskonomy_variant: str = 'tiny'
        tasks: List[str] = field(default_factory=lambda: ['rgb'])
        datasets: List[str] = field(default_factory=lambda: ['taskonomy', 'replica', 'gso'])
        transform: Optional[Union[Dict[str, Callable], str]] = "DEFAULT"  # List[Transform], None, "DEFAULT"
        image_size: Optional[int] = None
        num_positive: Union[int, str] = 1 # Either int or 'all'
        normalize_rgb: bool = False
        normalization_mean: List[float] = field(default_factory=lambda: [0.5,0.5,0.5])
        normalization_std: List[float] = field(default_factory=lambda: [0.5,0.5,0.5])
        force_refresh_tmp: bool = False


    def load_datasets(self, options):
        # Load saved image locations if they exist, otherwise create and save them
        self.urls = defaultdict(list)
        self.size = 0

        for dataset in self.datasets:
            
            if dataset == 'taskonomy':
                tmp_path = './tmp/{}_{}_{}.pkl'.format(
                    dataset,
                    '-'.join(options.tasks), 
                    f'{options.taskonomy_variant}-{options.split}'
                )
            else:
                tmp_path = './tmp/{}_{}_{}.pkl'.format(
                    dataset,
                    '-'.join(options.tasks), 
                    options.split
                )

            tmp_exists = os.path.exists(tmp_path)
            print("data temp path : ", tmp_path)

            if tmp_exists and not self.force_refresh_tmp:
                with open(tmp_path, 'rb') as f:
                    dataset_urls = pickle.load(f)
                    for task, urls in dataset_urls.items():
                        if task not in options.tasks: continue
                        # TODO remove later
                        task2 = 'segment_semantic' if task == 'segment_panoptic' else task
                        self.urls[task2] += urls
                dataset_size = len(dataset_urls[self.tasks[0]])
                self.size += dataset_size
                print(f'Loaded {dataset} with {dataset_size} images from tmp.')

            else:
                
                if dataset == 'taskonomy':
                    dataset_urls = {task: make_taskonomy_dataset(
                                            os.path.join(self.taskonomy_data_path, task), task, self.taskonomy_buildings)
                                        for task in options.tasks}
                elif dataset == 'replica':
                    dataset_urls = {task: make_replica_gso_dataset(
                                            self.replica_data_path, task, self.replica_buildings)
                                        for task in options.tasks}
                elif dataset == 'gso':
                    dataset_urls = {task: make_replica_gso_dataset(
                                             self.gso_data_path, task, self.gso_buildings) 
                                        for task in options.tasks}
                elif dataset == 'hypersim':
                    # dataset_urls = {task: make_hypersim_dataset(
                    #                          self.hypersim_data_path, task, self.hypersim_buildings) 
                    #                     for task in options.tasks}
                    dataset_urls = {task: make_hypersim_dataset_orig_split(
                                             self.hypersim_data_path, task, self.split) 
                                        for task in options.tasks}

                elif dataset == 'blendedMVS':
                    dataset_urls = {task: make_blendedMVS_dataset(
                                             self.blendedMVS_data_path, task, self.blendedMVS_buildings) 
                                        for task in options.tasks}

                dataset_urls, dataset_size  = self._remove_unmatched_images(dataset_urls)

                for task, urls in dataset_urls.items():
                    self.urls[task] += urls
                self.size += dataset_size

                # Save extracted URLs
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                with open(tmp_path, 'wb') as f:
                    pickle.dump(dataset_urls, f)

        
    def __init__(self, options: Options):
        start_time = perf_counter()
        
        if isinstance(options.tasks, str):
            options.tasks = [options.tasks]
            options.transform = {options.tasks: options.transform}        
        
        self.taskonomy_data_path = options.taskonomy_data_path
        self.replica_data_path = options.replica_data_path
        self.gso_data_path = options.gso_data_path
        self.hypersim_data_path = options.hypersim_data_path
        self.blendedMVS_data_path = options.blendedMVS_data_path
        self.datasets = options.datasets
        self.split = options.split
        self.image_size = options.image_size
        self.tasks = options.tasks
        self.num_positive = MAX_VIEWS if options.num_positive == 'all' else options.num_positive
        self.normalize_rgb = options.normalize_rgb
        self.normalization_mean = options.normalization_mean
        self.normalization_std = options.normalization_std
        self.force_refresh_tmp = options.force_refresh_tmp

        self.taskonomy_buildings = taskonomy_flat_split_to_buildings[f'{options.taskonomy_variant}-{self.split}']
        self.replica_buildings = replica_flat_split_to_buildings[self.split]
        self.gso_buildings = gso_flat_split_to_buildings[self.split]
        self.hypersim_buildings = hypersim_flat_split_to_buildings[self.split]
        self.blendedMVS_buildings = blendedMVS_flat_split_to_buildings[self.split]

        # ### taskonomy filtering
        # taskonomy_filtered_bpv = filter_taskonomy()
        # taskonomy_train_filter_thresh = 50000
        # taskonomy_val_filter_thresh = 100000
        # self.taskonomy_train_filter = taskonomy_filtered_bpv[taskonomy_train_filter_thresh]
        # self.taskonomy_val_filter = taskonomy_filtered_bpv[taskonomy_val_filter_thresh]


        # ### blendedMVS filtering
        # blendedMVS_filtered_bpv = filter_blendedMVS()
        # blendedMVS_train_filter_thresh = 10000
        # blendedMVS_val_filter_thresh = 20000
        # self.blendedMVS_train_filter = blendedMVS_filtered_bpv[blendedMVS_train_filter_thresh]
        # self.blendedMVS_val_filter = blendedMVS_filtered_bpv[blendedMVS_val_filter_thresh]

        self.load_datasets(options)
        self.transform = options.transform
        if isinstance(self.transform, str):
            if self.transform == 'DEFAULT':
                self.transform = {task: get_transform(task, image_size=None) for task in self.tasks}
            else:
                raise ValueError('TaskonomyDataset option transform must be a Dict[str, Callable], None, or "DEFAULT"')
                
        if self.normalize_rgb and 'rgb' in self.transform:
            self.transform['rgb'] = transforms.Compose(
                self.transform['rgb'].transforms +
                [transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)]

            )
    
        # Saving some lists and dictionaries for fast lookup

        self.tbpv_dict = {} # Save task -> building -> point -> view dict
        self.url_dict = {}  # Save (task, building, point, view) -> URL dict
        self.bpv_count = {} # Dictionary to check if all (building, point, view) tuples have all tasks
        
        for task in self.tasks:
            self.tbpv_dict[task] = {}
            filter_count = 0
            for url in self.urls[task]:
                if url.__contains__('replica-taskonomized'):
                    building = url.split('/')[-3]
                elif url.__contains__('replica-google-objects'): # e.g apartment_0-3, apartment_0-6
                    building = url.split('/')[-4] + '-' + url.split('/')[-3]
                elif url.__contains__('hypersim'): # e.g ai_001_001-cam_00, ai_001_001-cam_01
                    building = url.split('/')[-5] + '-' + url.split('/')[-3]
                elif url.__contains__('taskonomy'):
                    building = url.split('/')[-2]
                elif url.__contains__('BlendedMVS'):
                    building = url.split('/')[-3]
                else:
                    raise NotImplementedError('Dataset path (url) not recognized!')
                
                file_name = url.split('/')[-1].split('_')
  
                point, view = file_name[1], file_name[3]

                ##################
                # filter Taskonomy
                ##################
                # if building == 'grace' or building == 'rough': continue
                # if self.split == 'train':
                #     if building in self.taskonomy_train_filter.keys() and int(point) in self.taskonomy_train_filter[building].keys() and int(view) in self.taskonomy_train_filter[building][int(point)]: 
                #         filter_count += 1
                #         continue
                # elif self.split in ['val', 'test']:
                #     if building in self.taskonomy_val_filter.keys() and int(point) in self.taskonomy_val_filter[building].keys() and int(view) in self.taskonomy_val_filter[building][int(point)]: 
                #         filter_count += 1
                #         continue

                ##################
                # filter BlendedMVS
                ##################                
                if building in blendedMVS_bad_scenes: continue
                # if self.split == 'train':
                #     if building in self.blendedMVS_train_filter.keys() and int(point) in self.blendedMVS_train_filter[building]: 
                #         filter_count += 1
                #         continue
                # elif self.split in ['val', 'test']:
                #     if building in self.blendedMVS_val_filter.keys() and int(point) in self.blendedMVS_val_filter[building]: 
                #         filter_count += 1
                #         continue
                    

                # Populate url_dict
                self.url_dict[(task, building, point, view)] = url

                # Populate tbpv_dict
                if building not in self.tbpv_dict[task]:
                    self.tbpv_dict[task][building] = {}
                if point not in self.tbpv_dict[task][building]:
                    self.tbpv_dict[task][building][point] = []
                self.tbpv_dict[task][building][point].append(view)

                # Populate bpv_count
                if (building, point, view) not in self.bpv_count:
                    self.bpv_count[(building, point, view)] = 1
                else:
                    self.bpv_count[(building, point, view)] += 1
            
            # print(f"Task : {task} - split : {self.split} - filtered {filter_count} images!")


        # Remove entries that don't have all tasks and create list of all (building, point, view) tuples that contain all tasks
        self.bpv_list = [bpv_tuple for bpv_tuple, count in self.bpv_count.items() if count == len(self.tasks)]

        self.views = {}    # Build dictionary that contains all the views from a certain (building, point) tuple
        self.bpv_dict = {} # Save building -> point -> view dict
        for building, point, view in self.bpv_list:
            # Populate views
            if (building, point) not in self.views:
                self.views[(building, point)] = []
            self.views[(building, point)].append(view)

            # Populate bpv_dict
            if building not in self.bpv_dict:
                self.bpv_dict[building] = {}
            if point not in self.bpv_dict[building]:
                self.bpv_dict[building][point] = []
            self.bpv_dict[building][point].append(view)

            
        random.shuffle(self.bpv_list)
        
        end_time = perf_counter()
        self.num_points = len(self.views)
        self.num_images = len(self.bpv_list)
        self.num_buildings = len(self.bpv_dict)
        
        logger = logging.getLogger(__name__)
        logger.warning("Loaded {} images in {:0.2f} seconds".format(self.num_images, end_time - start_time))
        logger.warning("\t ({} buildings) ({} points) ({} images) for domains {}".format(self.num_buildings, self.num_points, self.num_images, self.tasks))


    def __len__(self):
        return len(self.bpv_list)

    def __getitem__(self, index):
        flip = random.random() > 0.5 
        result = {}
        
        # Anchor building / point / view
        building, point, view = self.bpv_list[index]
        
        positive_views = [view]
        positive_samples = {}   

        for task_num, task in enumerate(self.tasks):
            resize_method = Image.BILINEAR if task in ['rgb'] else Image.NEAREST
            task_samples = []
            for v in positive_views:
                path = self.url_dict[(task, building, point, v)]
                res = default_loader(path)

                if self.transform is not None and self.transform[task] is not None:

                    if path.__contains__('hypersim') or path.__contains__('BlendedMVS'):
                        resize_transform = transforms.Resize(self.image_size, resize_method)
                        res = resize_transform(res)
                        if task_num == 0: 
                            i, j, h, w = transforms.RandomCrop.get_params(
                                res, output_size=(self.image_size, self.image_size))
                        res = TF.crop(res, i, j, h, w)
                        res = self.transform[task](res)

                    else:
                        transform = transforms.Compose([
                            transforms.Resize(self.image_size, resize_method),
                            transforms.CenterCrop(self.image_size),
                            self.transform[task]])
                        res = transform(res)
                    

                # flip augmentation
                if flip: 
                    res = torch.flip(res, [2])
                    if task == 'normal': res[0,:,:] = 1 - res[0,:,:]

                task_samples.append(res)

            task_samples = torch.stack(task_samples) if self.num_positive > 1 else task_samples[0]

            positive_samples[task] = task_samples
        
        positive_samples['point'] = point
        positive_samples['building'] = building
        result['positive'] = positive_samples
        
        
        return result
            


    def randomize_order(self, seed=0):
        random.seed(0)
        random.shuffle(self.bpv_list)
    
    def task_config(self, task):
        return task_parameters[task]

    def _remove_unmatched_images(self, dataset_urls) -> (Dict[str, List[str]], int):
        '''
            Filters out point/view/building triplets that are not present for all tasks
            
            Returns:
                filtered_urls: Filtered Dict
                max_length: max([len(urls) for _, urls in filtered_urls.items()])
        '''
        n_images_task = [(len(obs), task) for task, obs in dataset_urls.items()]
        max_images = max(n_images_task)[0]
        if max(n_images_task)[0] == min(n_images_task)[0]:
            return dataset_urls, max_images
        else:
            print("Each task must have the same number of images. However, the max != min ({} != {}). Number of images per task is: \n\t{}".format(
                max(n_images_task)[0], min(n_images_task)[0], "\n\t".join([str(t) for t in n_images_task])))
            # Get views for each task
            def _parse_fpath_for_view( path ):
                url = path
                if url.__contains__('replica-taskonomized'):
                    building = url.split('/')[-3]
                elif url.__contains__('replica-google-objects'): # e.g apartment_0-3, apartment_0-6
                    building = url.split('/')[-4] + '-' + url.split('/')[-3]
                elif url.__contains__('hypersim'): # e.g ai_001_001-cam_00, ai_001_001-cam_01
                    building = url.split('/')[-5] + '-' + url.split('/')[-3]
                elif url.__contains__('taskonomy'):
                    building = url.split('/')[-2]
                elif url.__contains__('BlendedMVS'):
                    building = url.split('/')[-3]
                # building = os.path.basename(os.path.dirname(path))
                file_name = os.path.basename(path) 
                lf = parse_filename( file_name )
                return View(view=lf.view, point=lf.point, building=building)

            self.task_to_view = {}
            for task, paths in dataset_urls.items():
                self.task_to_view[task] = [_parse_fpath_for_view( path ) for path in paths]
    
            # Compute intersection
            intersection = None
            for task, uuids in self.task_to_view.items():
                if intersection is None:
                    intersection = set(uuids)
                else:
                    intersection = intersection.intersection(uuids)
            # Keep intersection
            print('Keeping intersection: ({} images/task)...'.format(len(intersection)))
            new_urls = {}
            for task, paths in dataset_urls.items():
                new_urls[task] = [path for path in paths if _parse_fpath_for_view( path ) in intersection]
            return new_urls, len(intersection)
        raise NotImplementedError('Reached the end of this function. You should not be seeing this!')



def make_taskonomy_dataset(dir, task, folders=None):
    # TODO remove later
    if task == 'segment_semantic': dir = os.path.join(dir, '..', 'segment_panoptic')
    #  folders are building names. If None, get all the images (from both building folders and dir)
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    for subfolder in folders:
        subfolder_path = os.path.join(dir, subfolder)
        print(subfolder_path)
        if os.path.isdir(subfolder_path) and (folders is None or subfolder in folders):
            for fname in sorted(os.listdir(subfolder_path)):
                path = os.path.join(subfolder_path, fname)
                images.append(path)

        # If folders/buildings are not specified, use images in dir
        if folders is None and os.path.isfile(subfolder_path):
            images.append(subfolder_path)

    return images

def make_replica_gso_dataset(dir, task, folders=None):
    if task == 'segment_semantic': task = 'semantic'
    #  folders are building names. 
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"
    
    if folders is None:
        folders = os.listdir(dir)

    for folder in folders:
        if folder not in REPLICA_BUILDINGS: # gso dataset e.g. apartment_0-3, apartment_0-6, apartment_0-15
            folder_path = os.path.join(dir, folder.split('-')[0], folder.split('-')[1], task)
        else: # replica dataset
            folder_path = os.path.join(dir, folder, task)
        for fname in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, fname)
            images.append(path)

    return images

def make_hypersim_dataset(dir, task, folders=None):
    if task == 'segment_semantic': task = 'semantic_hdf5'
    #  folders are building names. 
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    for folder in folders:
        taskonomized_path = os.path.join(dir, folder, 'taskonomized')
        for camera in os.listdir(taskonomized_path):
            if not camera.startswith('cam'):
                continue
            # filter out bad points from filtered_points.json
            with open(os.path.join(taskonomized_path, camera, 'filtered_points.json')) as json_file:
                bad_points = json.load(json_file)
            folder_path = os.path.join(taskonomized_path, camera, task)
            for fname in sorted(os.listdir(folder_path)):
                point = fname.split('_')[1]
                if point not in bad_points:
                    path = os.path.join(folder_path, fname)
                    images.append(path)

    return images

def make_hypersim_dataset_orig_split(dir, task, split):
    hypersim_orig_split_file = os.path.join(os.path.dirname(__file__), 'splits', f'{split}_hypersim_orig.csv')
    df = pd.read_csv(hypersim_orig_split_file)

    if task == 'segment_semantic': task = 'semantic_hdf5'
    if task in ['normal', 'mask_valid', 'depth_zbuffer']: task = task + '2'
    #  folders are building names. 
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"

    folders = [scene for scene in set(df['scene_name'].tolist()) if scene in os.listdir(dir)]
    for folder in folders:
        scene = df.loc[df['scene_name']==folder]
        taskonomized_path = os.path.join(dir, folder, 'taskonomized')
        for camera in os.listdir(taskonomized_path):
            if not camera.startswith('cam'):
                continue
            scene_cam = scene.loc[scene['camera_name']==camera]
            # filter out bad points from filtered_points.json
            # with open(os.path.join(taskonomized_path, camera, 'filtered_points.json')) as json_file:
            #     bad_points = json.load(json_file)
            folder_path = os.path.join(taskonomized_path, camera, task)
            if not os.path.exists(folder_path): continue
            for fname in sorted(os.listdir(folder_path)):
                point = fname.split('_')[1]
                # if point in bad_points: continue
                try:
                    row = scene_cam.loc[scene_cam['frame_id']==int(point)]
                except:
                    continue
                if not row.empty and row.iloc[0]['included_in_public_release'] and row.iloc[0]['split_partition_name'] == split:
                    path = os.path.join(folder_path, fname)
                    images.append(path)

        print(len(images))

    return images

def make_blendedMVS_dataset(dir, task, folders=None):
    if task == 'segment_semantic': task = 'semantic'
    if task == 'mask_valid': task = 'mask_valid_depth'
    #  folders are building names. 
    images = []
    dir = os.path.expanduser(dir)
    if not os.path.isdir(dir):
        assert "bad directory"
    
    if folders is None:
        folders = os.listdir(dir)

    for folder in folders:
        folder_path = os.path.join(dir, folder, task)
        for fname in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, fname)
            images.append(path)

    return images


def make_empty_like(data_dict):
    if not isinstance(data_dict, dict):
        if isinstance(data_dict, torch.Tensor):
            return torch.zeros_like(data_dict)
        elif isinstance(data_dict, list):
            return [make_empty_like(d) for d in data_dict]
        else:
            return type(data_dict)()
        raise NotImplementedError

    result = {}
    for k, v in data_dict.items():
        result[k] = make_empty_like(v)
    return result

