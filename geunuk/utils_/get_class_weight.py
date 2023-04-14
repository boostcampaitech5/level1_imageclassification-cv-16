import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def calc_class_weight(train_dataset, model_name, num_classes, device):
    if model_name == "Mask":
        mask_labels = [mask_label for mask_label in train_dataset.mask_labels]
        mask_labels.sort()

        mask_labels_weight = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=mask_labels)
        mask_labels_weight = torch.FloatTensor(mask_labels_weight).to(device)
        
        return mask_labels_weight
    
    elif model_name == "Gender":
        # -- Class Weight Calculate
        gender_labels = [gender_label for gender_label in train_dataset.gender_labels]
        gender_labels.sort()

        gender_labels_weight = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=gender_labels)
        gender_labels_weight = torch.FloatTensor(gender_labels_weight).to(device)
        
        return gender_labels_weight
    
    elif model_name == "Age":
        # -- Class Weight Calculate
        age_labels = [age_label for age_label in train_dataset.age_labels]
        age_labels.sort()

        age_labels_weight = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=age_labels)
        age_labels_weight = torch.FloatTensor(age_labels_weight).to(device)
        
        return age_labels_weight