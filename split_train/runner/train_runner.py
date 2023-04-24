import wandb
import torch
import numpy as np
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

class CustomTrainer():
    def __init__(self, args, model, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, device):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device 
    
    def train(self, model_name):
        self.model.to(self.device)

        best_val_loss = np.inf
        best_val_score = 0
        best_model = None

        # Early Stop
        patience_limit = self.args.patience
        patience = 0

        for epoch in range(1, self.args.epochs+1):
            self.model.train()
            train_loss = []
            
            lr = self.optimizer.param_groups[0]['lr']
            print('='*25, f'TRAIN epoch:{epoch}', '='*25, f'lr:{lr:.9f}')
            
            for imgs, labels in tqdm(iter(self.train_dataloader)):
                imgs = imgs.float().to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(imgs)
                loss = self.criterion(output, labels)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

            _val_loss, _val_acc = self.validation()
            _train_loss = np.mean(train_loss)

            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')
            print(' ')
            wandb.log({f"Epoch": epoch, f"{model_name}/Train_Loss": _train_loss, f"{model_name}/Val_Loss": _val_loss, f"{model_name}/Val_ACC": _val_acc})

            if self.scheduler is not None:
                self.scheduler.step(_val_loss)

            if best_val_loss > _val_loss:
                best_val_loss = _val_loss
                best_val_score = _val_acc
                best_model = self.model
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    break

        print(f'Best Loss : [{best_val_loss:.5f}] Best ACC : [{best_val_score:.5f}]')
        return best_model, best_val_score, best_val_loss

    def validation(self):
        self.model.eval()
        val_loss = []
        preds, trues = [], []
        print('='*25, f'VALID', '='*25)
        with torch.no_grad():
            for imgs, labels in tqdm(iter(self.valid_dataloader)):
                imgs = imgs.float().to(self.device)
                labels = labels.to(self.device)

                logit = self.model(imgs)

                loss = self.criterion(logit, labels)

                val_loss.append(loss.item())

                preds += logit.argmax(1).detach().cpu().numpy().tolist()
                trues += labels.detach().cpu().numpy().tolist()

            _val_loss = np.mean(val_loss)
            _val_acc = f1_score(trues, preds, average='macro')

        return _val_loss, _val_acc