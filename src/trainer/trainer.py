import os
import time
from tempfile import TemporaryDirectory
import warnings
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import wandb
from tqdm import tqdm

import torch

class Trainer:
    def __init__(self, model, dataloaders, loss_fn, optimizer, scheduler, device, metric, num_epochs=10, wandb_log=False):
        self.model = model
        self.train_dataloader = dataloaders['train']
        self.test_dataloader = dataloaders['validation']
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.metric = metric
        self.num_epochs = num_epochs
        self.wandb_log = wandb_log

    def train_epochs(self, num_epochs=None):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)

            train_loss = self.train()
            test_loss, metric = self.test()
            self.scheduler.step()
            print(f"train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, {self.metric}: {metric:.4f}\n")
            if self.wandb_log:
                wandb.log({'train_loss': train_loss, 'valid_loss': test_loss, self.metric: metric})

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Final {self.metric}: {metric:4f}\n')
        return self.model, metric
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return
    
    def train(self):
        num_batches = len(self.train_dataloader)
        self.model.train()
        
        train_loss = 0
        for X, y in tqdm(self.train_dataloader, total=num_batches):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= num_batches
        return train_loss

    def test(self):
        num_batches = len(self.test_dataloader)
        self.model.eval()
        
        test_loss = 0
        metric = 0
        with torch.no_grad():
            for X, y in tqdm(self.test_dataloader, total=num_batches):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                _, preds = torch.max(outputs, 1)
                test_loss += self.loss_fn(outputs, y).item()
                if self.metric == 'balanced_accuracy':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UserWarning)
                        metric += balanced_accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
        test_loss /= num_batches
        metric /= num_batches
        return test_loss, metric