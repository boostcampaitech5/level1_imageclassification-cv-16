import os
import cv2
import random
import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import train_test_split

def get_transforms(img_size, transforms):
    if transforms:
         return A.Compose([
                    # A.CenterCrop(height=375, width=200, p=1.0),
                    A.Resize(img_size[0], img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                    ToTensorV2()
                    ])
    else:
        return A.Compose([
                    # A.CenterCrop(height=375, width=200, p=1.0),
                    A.Resize(img_size[0], img_size[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                    ToTensorV2()
                    ])
        
class CustomTrainDataset(Dataset):
    _mask_labels = {"mask1": 0, "mask2": 0, "mask3": 0, "mask4": 0, "mask5": 0, "incorrect_mask": 1, "normal": 2}
    _gender_labels = {"male": 0, "female": 1}

    def __init__(self, model_name, data_dir, folder_list, resize, transforms):
        self.model_name = model_name
        self.data_dir = data_dir
        self.folder_list = folder_list
        self.resize = resize
        self.transforms = transforms
        
        self.image_paths = []
        self.mask_labels, self.gender_labels, self.age_labels = [], [], []
        self.setup()
        
    def setup(self):
        for folder in self.folder_list:
            if folder.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue
            img_folder = os.path.join(self.data_dir, folder) # ('/workspace/data/train/image', 000004_male_Asian_54)
            for file_name in os.listdir(img_folder): # ('mask1.jpg', 'mask2.jpg', ... )
                if file_name.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                    continue
                _file_name, _ = os.path.splitext(file_name) # ('mask1', '.jpg')
                img_path = os.path.join(self.data_dir, folder, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                
                _, gender, _, age = folder.split("_")
                mask_label = self._mask_labels[_file_name]                
                gender_label = self._gender_labels[gender]
                age = int(age)
                age_label = 0 if age < 30 else 1 if age < 60 else 2
                
                self.image_paths.append(img_path)
                if self.model_name == 'Mask': self.mask_labels.append(mask_label)
                elif self.model_name == 'Gender': self.gender_labels.append(gender_label)
                elif self.model_name == 'Age': self.age_labels.append(age_label)

    def __getitem__(self, index):
        assert self.transforms is not None, "transform 을 주입해주세요"
        
        image = cv2.imread(self.image_paths[index])
        trfm = get_transforms(self.resize, self.transforms)
        image = trfm(image=image)['image']
        
        if self.model_name == 'Mask':
            mask_label = self.mask_labels[index]
            return image, mask_label
        elif self.model_name == 'Gender':
            gender_label = self.gender_labels[index]
            return image, gender_label
        elif self.model_name == 'Age':
            age_label = self.age_labels[index]
            return image, age_label
        
    def __len__(self):
        return len(self.image_paths)
    
    
class CustomTrainDataset_60(Dataset):
    _mask_labels = {"mask1": 0, "mask2": 0, "mask3": 0, "mask4": 0, "mask5": 0, "incorrect_mask": 1, "normal": 2}
    _gender_labels = {"male": 0, "female": 1}

    def __init__(self, model_name, data_dir, folder_list, resize, transforms):
        self.model_name = model_name
        self.data_dir = data_dir
        self.folder_list = folder_list
        self.resize = resize
        self.transforms = transforms
        self.age_transforms = A.Compose([
                    A.Resize(resize[0], resize[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                    A.GaussNoise(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.ColorJitter(p=0.5),
                    ToTensorV2()
                    ])

        self.image_paths = []
        self.mask_labels, self.gender_labels, self.age_labels = [], [], []
        self.cnt_60 = 0
        self.setup()

    def setup(self):
        for folder in self.folder_list:
            if folder.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue
            img_folder = os.path.join(self.data_dir, folder) # ('/workspace/data/train/image', 000004_male_Asian_54)
            for file_name in os.listdir(img_folder): # ('mask1.jpg', 'mask2.jpg', ... )
                if file_name.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                    continue
                _file_name, _ = os.path.splitext(file_name) # ('mask1', '.jpg')
                img_path = os.path.join(self.data_dir, folder, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
 
                _, gender, _, age = folder.split("_")
                if self.model_name == 'Gender':
                    miss_label = ['006364','006363','006362','006361','006360','006359','004432','001720','001498_1']
                    if id in miss_label:
                        if gender == 'male': gender == 'feamale'
                        elif gender == 'female': gender = 'male'
                mask_label = self._mask_labels[_file_name]                
                gender_label = self._gender_labels[gender]
                age = int(age)
                age_label = 0 if age < 30 else 1 if age < 60 else 2
                self.image_paths.append(img_path)
                self.age_labels.append(age_label)

                if self.model_name=='Age':
                    for _ in range(2): # 변경 하면 원하는 만큼 데이터 추가 가능
                        if self.model_name=='Age':
                            if age_label == 2:
                                self.cnt_60 += 1
                                image = cv2.imread(self.image_paths[self.image_paths.index(img_path)])
                                image = self.age_transforms(image=image)['image']
                                self.image_paths.append(image) # 이미지를 넣음
                                self.age_labels.append(age_label)
                            # image_paths 길이 두배, age_labels길이도 두배

                elif self.model_name == 'Mask': self.mask_labels.append(mask_label)
                elif self.model_name == 'Gender': self.gender_labels.append(gender_label)


    def __getitem__(self, index):
        assert self.transforms is not None, "transform 을 주입해주세요"

        if self.model_name == 'Age':
            if type(self.image_paths[index]) is str:
                image = cv2.imread(self.image_paths[index])
                trfm = get_transforms(self.resize, self.transforms)
                image = trfm(image=image)['image']
            else:
                image = self.image_paths[index]
            age_label = self.age_labels[index]
            return image, age_label
        
        image = cv2.imread(self.image_paths[index])
        trfm = get_transforms(self.resize, self.transforms)
        image = trfm(image=image)['image']
        
        if self.model_name == 'Mask':
            mask_label = self.mask_labels[index]
            return image, mask_label
        elif self.model_name == 'Gender':
            gender_label = self.gender_labels[index]
            return image, gender_label
        
    def __len__(self):
        return len(self.age_labels)
