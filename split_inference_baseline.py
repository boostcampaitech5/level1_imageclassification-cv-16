import argparse
import multiprocessing
import os
from importlib import import_module

from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from split_dataset_baseline import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def inference(img_paths, model_dir, args, model_name, num_classes):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
 
    model_dir = os.path.join(model_dir, model_name)
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print(f"{model_name} calculating inference results..")
    preds = []
    with torch.no_grad():
        for images in tqdm(iter(loader), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
            
    return preds

@torch.no_grad()
def valid_inference(data_dir, model_dir, args, model_name, num_classes):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # -- dataset
    dataset_module = getattr(import_module("dataset_3head"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        model_name=model_name
    )
    
    # -- augmentation
    transform_module = getattr(import_module("dataset_3head"), 'BaseAugmentation')  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    _, val_set = dataset.split_dataset()
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    model_dir = os.path.join(model_dir, model_name)
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    print(f"{model_name} calculating Valid results..")
    preds_list, labels_list = [], []
    with torch.no_grad():
        for val_batch in tqdm(iter(val_loader), bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            preds = model(inputs)
            preds = torch.argmax(preds, dim=-1)
            
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            
    return preds_list, labels_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', nargs="+", type=int, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--train_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    
    #-- Mask Inference
    model_name = 'mask'
    mask_preds = inference(img_paths, model_dir, args, model_name, num_classes=3)
    
    #-- Gender Inference
    model_name = 'gender'
    gender_preds = inference(img_paths, model_dir, args, model_name, num_classes=2)
    
    #-- Age Inference
    model_name = 'age'
    age_preds = inference(img_paths, model_dir, args, model_name, num_classes=3)
    
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
    
    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")
    print(" ")
    print("Validation Inference Start !!")
    model_name = 'mask'
    mask_preds, mask_labels = valid_inference(args.train_data_dir, model_dir, args, model_name, num_classes=3)
    model_name = 'gender'
    gender_preds, gender_labels = valid_inference(args.train_data_dir, model_dir, args, model_name, num_classes=2)
    model_name = 'age'
    age_preds, age_labels = valid_inference(args.train_data_dir, model_dir, args, model_name, num_classes=3)
    
    preds, labels = [], []
    for (mask_pred, gender_pred, age_pred), (mask_label, gender_label, age_label) in zip(zip(mask_preds, gender_preds, age_preds), zip(mask_labels, gender_labels, age_labels)):
        pred = (mask_pred, gender_pred, age_pred)
        preds.append(label_dict[pred])
        label = (mask_label, gender_label, age_label)
        labels.append(label_dict[label])
        
    print('3Head F1 score : ', f1_score(labels, preds, average='macro'))
