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
from utils_.sam import SAM
from runner.pytorch_timm import TimmModel
from runner.train_runner import CustomTrainer
from data.dataset import CustomTrainDataset


def main(args, train_df, data_dir, model_name, num_classes):
    gc.collect() # python 자원 관리 
    torch.cuda.empty_cache() # gpu 자원관리

    if model_name == 'Gender':
        drop_label = ['006424', '006339', '003713', '003437', '003421', '003399', '003294', '003169', '003113', '003014', '001520', '001266', '000725', '000647']
        drop_label_index = [train_df[train_df['id']==dl].index[0] for dl in drop_label]
        for index in drop_label_index:
            train_df = train_df.drop(index)            
        train_, val_ = train_test_split(train_df, test_size=0.2, random_state=args.seed, stratify=train_df['gender_label'])
    elif model_name == 'Age':
        train_, val_ = train_test_split(train_df, test_size=0.2, random_state=args.seed, stratify=train_df['age_label'])
    else:
        train_, val_ = train_test_split(train_df, test_size=0.2, random_state=args.seed)

    print('='*25, f'{model_name} Model Train Start', '='*25)
    
    # -- dataset & dataloader
    train_dataset = CustomTrainDataset(model_name, data_dir, train_['path'].values, args.resize, transforms=True)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=8)
    
    val_dataset = CustomTrainDataset(model_name, data_dir, val_['path'].values, args.resize, transforms=False)
    valid_dataloader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=8)

    model = TimmModel(args, num_classes=num_classes, pretrained=True).to(device)
    model = nn.DataParallel(model)
    
    class_weight = calc_class_weight(train_dataset, model_name, num_classes, device)
    # criterion = FocalLoss(weight=class_weight).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    
    base_optim = torch.optim.AdamW 
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    # optimizer = SAM(model.parameters(), base_optim, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs',min_lr=1e-9, verbose=True)

    trainer = CustomTrainer(args, model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device)
    model, best_score, best_loss = trainer.train(model_name)
    
    print('='*25, f'{model_name} Model Train End', '='*25)
    
    return model, best_score, best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='resnet50', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='CrossEntropy', help='criterion type CrossEntropy|FocalLoss|F1Loss|LabelSmoothingLoss (default: cross_entropy)')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type = int, default = 5, help = 'patience for earlystopping')
    parser.add_argument('--how', default='Split', help='Model split mask, gender, age')
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    data_dir = TRAIN_IMG_FOLDER_PATH
    model_dir = '/opt/ml/level1_imageclassification-cv-16/geunuk/models'    
    project_idx = len(glob(os.path.join(model_dir, '*'))) + 1
    os.makedirs(os.path.join(model_dir, f'Project{project_idx}'), exist_ok=True)
         
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Mask_Classification",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": args.model,
        "epochs": args.epochs,
        "batch_size" : args.batch_size,
        "loss" : args.criterion,
        "how": args.how,
        },
        name=f"{args.model}_{args.criterion}_resize{args.resize}_Project[{project_idx}]_lr[{args.lr}]"
    )
    
    #-- Data Load & Labeling
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['gender_label'], train_df['age_label'] = None, None

    _gender_labels = {"male": 0, "female": 1}
    train_df['gender_label'] = train_df['path'].apply(lambda x : _gender_labels[str(x).split('_')[1]])
    
    train_df['age_label'] = train_df['path'].apply(lambda x : int((str(x).split('_')[3])))
    train_df['age_label'].loc[train_df['age_label'] < 25] = 0
    train_df['age_label'].loc[(train_df['age_label'] >= 25) & (train_df['age_label'] < 55)] = 1
    train_df['age_label'].loc[train_df['age_label'] >= 55] = 2
    
    model_name = "Mask"
    mask_model, mask_best_score, mask_best_loss = main(args, train_df, data_dir, model_name, 3)    
    torch.save(mask_model.module.state_dict(), f'{model_dir}/Project{project_idx}/{model_name}_[{args.model}]_[score{mask_best_score:.4f}]_[loss{mask_best_loss:.4f}].pt')
    
    model_name = "Gender"
    gender_model, gender_best_score, gender_best_loss = main(args, train_df, data_dir, model_name, 2)
    torch.save(gender_model.module.state_dict(), f'{model_dir}/Project{project_idx}/{model_name}_[{args.model}]_[score{gender_best_score:.4f}]_[loss{gender_best_loss:.4f}].pt')
    
    model_name = "Age"
    age_model, age_best_score, age_best_loss = main(args, train_df, data_dir, model_name, 3)
    torch.save(age_model.module.state_dict(), f'{model_dir}/Project{project_idx}/{model_name}_[{args.model}]_[score{age_best_score:.4f}]_[loss{age_best_loss:.4f}].pt')
    
    