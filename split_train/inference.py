import warnings
warnings.filterwarnings(action='ignore')

import os
import gc
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import wandb

from utils_.set_path import *
from utils_.set_seed import seed_everything
from utils_.loss import FocalLoss, LabelSmoothingLoss, F1Loss
from utils_.get_class_weight import calc_class_weight
from runner.pytorch_timm import TimmModel
from runner.train_runner import CustomTrainer
from data.dataset import CustomTestDataset

def main(args, model, image_paths, device):
    test_dataset = CustomTestDataset(image_paths, args.resize, transforms=False)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8)
    
    model.to(device)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_dataloader), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            imgs = imgs.to(device)
            
            logit = model(imgs)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            
    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='resnet50', help='model type (default: BaseModel)')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--how', default='Split', help='Model split mask, gender, age')
        
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    test_dir = TEST_IMG_PATH
    model_dir = '/opt/ml/level1_imageclassification-cv-16/geunuk/models'  
    project_idx = len(glob(os.path.join(model_dir, '*')))
    print(f'Project {project_idx} inference Start.')
    
    test_df = pd.read_csv(TEST_CSV_PATH)
    image_paths = [os.path.join(test_dir, img_id) for img_id in test_df.ImageID]
    
    # Mask
    num_classes = 3
    mask_model_weights = torch.load(glob(f'{model_dir}/Project{project_idx}/Mask*')[0])
    mask_model = TimmModel(args, num_classes=num_classes, pretrained=True).to(device)
    mask_model.load_state_dict(mask_model_weights)
    
    mask_preds = main(args, mask_model, image_paths, device)
    
    # Gender
    num_classes = 2
    gender_model_weights = torch.load(glob(f'{model_dir}/Project{project_idx}/Gender*')[0])
    gender_model = TimmModel(args, num_classes=num_classes, pretrained=True).to(device)
    gender_model.load_state_dict(gender_model_weights)
    
    gender_preds = main(args, gender_model, image_paths, device)
    
    # Age
    num_classes = 3
    age_model_weights = torch.load(glob(f'{model_dir}/Project{project_idx}/Age*')[0])
    age_model = TimmModel(args, num_classes=num_classes, pretrained=True).to(device)
    age_model.load_state_dict(age_model_weights)
    
    age_preds = main(args, age_model, image_paths, device)
    
    """
    Mask               Gender          Age
    - mask      : 0    - male   : 0    - 30 미만         : 0 
    - incorrect : 1    - female : 1    - 30 이상 60 미만 : 1
    - normal    : 2                    - 60 이상         : 2
    """
    label_dict = {(0, 0, 0): 0, (0, 0, 1): 1, (0, 0, 2): 2, (0, 1, 0): 3, (0, 1, 1): 4, 
                  (0, 1, 2): 5, (1, 0, 0): 6, (1, 0, 1): 7, (1, 0, 2): 8, (1, 1, 0): 9, 
                  (1, 1, 1): 10, (1, 1, 2): 11, (2, 0, 0): 12, (2, 0, 1): 13, (2, 0, 2): 14, 
                  (2, 1, 0): 15, (2, 1, 1): 16, (2, 1, 2): 17}
    
    preds = []
    for mask_pred, gender_pred, age_pred in zip(mask_preds, gender_preds, age_preds):
        temp = (mask_pred, gender_pred, age_pred)
        preds.append(label_dict[temp])
        
    test_df['ans'] = preds
    test_df.to_csv(os.path.join(SUBMIT_SAVE_PATH, f'{args.model}_Project{project_idx}.csv'), index=False)
    print(f'Project {project_idx} inference is done!')