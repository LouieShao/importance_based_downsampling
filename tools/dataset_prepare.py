import os
import sys
import warnings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
warnings.filterwarnings('ignore')

import argparse
import functools
import time

import torch
from torch.utils.data import DataLoader

from tools.scripts import train_classification, validate_classification
from tools.utils import (get_logger, set_seed, worker_seed_init_fn,
                         compute_flops_and_params, build_optimizer,
                         build_scheduler, build_training_mode)

import shutil
from label_livingornot import *
from tqdm import tqdm
import random
LIVING_NUM = len(living_keys)
NONLIVING_NUM = len(nonliving_keys)
from simpleAICV.classification import backbones
print('whaaa')
def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='LOCAL_PROCESS_RANK in DistributedDataParallel model')
    parser.add_argument(
        '--living',
        type=int,
        default=0,
        help='0: living 1: nonliving 2: random with living class num 3: random with nonliving class num')
    parser.add_argument(
        '--hint',
        type=str,
        default='',
        help='mark the iteration info')
    return parser.parse_args()

def dataset_prepare(living):
    print('preparing the dataset....')
    if os.path.isdir('/home/shaoshihao/imagenet_temp'):
        shutil.rmtree('/home/shaoshihao/imagenet_temp')
    if os.path.isdir('/home/shaoshihao/imagenet_temp_val'):
        shutil.rmtree('/home/shaoshihao/imagenet_temp_val')
    os.mkdir('/home/shaoshihao/imagenet_temp')
    os.mkdir('/home/shaoshihao/imagenet_temp_val')
    if living == 0:
        for i in tqdm(living_keys):
            shutil.copytree(os.path.join('/home/shaoshihao/train',i),'/home/shaoshihao/imagenet_temp/'+i)
            shutil.copytree(os.path.join('/home/shaoshihao/val',i),'/home/shaoshihao/imagenet_temp_val/'+i)
        
    elif living == 1:
        for i in tqdm(nonliving_keys):
            shutil.copytree(os.path.join('/home/shaoshihao/train',i),'/home/shaoshihao/imagenet_temp/'+i)
            shutil.copytree(os.path.join('/home/shaoshihao/val',i),'/home/shaoshihao/imagenet_temp_val/'+i)
    elif living == 2:
        choices_picked = random.sample([k for k in range(1000)],LIVING_NUM)
        choices_picked = np.array(choices_picked)
        np.save('living_choices.npy',choices_picked)
        
        for i in tqdm(choices_picked):
            shutil.copytree(os.path.join('/home/shaoshihao/train',i),'/home/shaoshihao/imagenet_temp/'+i)
            shutil.copytree(os.path.join('/home/shaoshihao/val',i),'/home/shaoshihao/imagenet_temp_val/'+i)
    elif living == 3:
        choices_picked_opp = np.load('living_choices')
        pool_classes = [k for k in range(1000)]
        choices_picked = [t for t in pool_classes if t not in choices_picked_opp]
        for i in tqdm(choices_picked):
            shutil.copytree(os.path.join('/home/shaoshihao/train',i),'/home/shaoshihao/imagenet_temp/'+i)
            shutil.copytree(os.path.join('/home/shaoshihao/val',i),'/home/shaoshihao/imagenet_temp_val/'+i)
def main():
    
    args = parse_args()
    
    sys.path.append(args.work_dir)
    dataset_prepare(args.living)
    

    return


if __name__ == '__main__':
    main()
