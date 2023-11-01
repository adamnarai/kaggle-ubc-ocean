import os
import time
from tempfile import TemporaryDirectory
import warnings
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import wandb

import torch

class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, wandb_log=False):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.wandb_log = wandb_log
        self.dataset_sizes = {x: len(self.dataloaders[x].dataset) for x in ['train', 'validation']}
    
    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs
        return

    def train(self, num_epochs=None):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model.state_dict(), best_model_params_path)
            best_balanced_acc = 0.0

            for epoch in range(self.num_epochs):
                print(f'Epoch {epoch}/{self.num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'validation']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                    else:
                        self.model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0
                    balanced_acc_list = []

                    # Iterate over data.
                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            balanced_acc_list.append(balanced_accuracy_score(labels.cpu().data, preds.cpu().data))
                    if phase == 'train':
                        self.scheduler.step()

                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                    epoch_balanced_acc = np.mean(balanced_acc_list)

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} balanced_acc: {epoch_balanced_acc:.4f}')

                    # deep copy the model
                    if phase == 'validation' and epoch_balanced_acc > best_balanced_acc:
                        best_balanced_acc = epoch_balanced_acc
                        torch.save(self.model.state_dict(), best_model_params_path)

                    if self.wandb_log and phase == 'validation':
                        wandb.log({'valid_loss': epoch_loss, 'balanced_accuracy': epoch_balanced_acc})

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val balanced_acc: {best_balanced_acc:4f}')

            # load best model weights
            self.model.load_state_dict(torch.load(best_model_params_path))
        return self.model, best_balanced_acc
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return