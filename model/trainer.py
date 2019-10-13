import os
import random

import torch
import torch.nn as nn
import apex

from torch.utils.data import DataLoader
from tqdm import tqdm

from optim import RAdam
from loss import soft_dice_loss, focal_cannab


class AvgMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1

    def __call__(self):
        if self.count:
            return self.sum / self.count
        return 0


class Trainer:
    def __init__(self, model, optimizer_params={}, loss_params={}, amp_params={}, device=None, n_jobs=0):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        lr = optimizer_params.get('lr', 1e-3)
        weight_decay = optimizer_params.get('weight_decay', 0)
        focal_weight = loss_params.get('focal_weight', 1)
        dice_weight = loss_params.get('dice_weight', 0)
        opt_level = amp_params.get('opt_level', 'O0')
        loss_scale = amp_params.get('loss_scale', None)

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.device = device
        self.n_jobs = n_jobs
        self.last_epoch = 0
        self.model = model.to(self.device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bn', 'bias']
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        self.optimizer = RAdam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

        self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer,
                                                         opt_level=opt_level, loss_scale=loss_scale, verbosity=1)

    def _train_epoch(self, train_dataloader):
        pbar = tqdm(desc=f'Train, epoch #{self.last_epoch}', total=len(train_dataloader))
        self.model.train()

        loss = AvgMeter()
        for images, masks in train_dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            probs = torch.sigmoid(logits)
            batch_loss = self.dice_weight * soft_dice_loss(probs, masks) + \
                         self.focal_weight * focal_cannab(probs, masks)

            self.optimizer.zero_grad()

            with apex.amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                    
            self.optimizer.step()

            loss.update(batch_loss.item())
            pbar.update(1)
            pbar.set_postfix({'loss': loss()})

    @torch.no_grad()
    def _test_epoch(self, test_dataloader):
        pbar = tqdm(desc=f'Test, epoch #{self.last_epoch}', total=len(test_dataloader))
        self.model.eval()

        loss = AvgMeter()
        for images, masks in test_dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            probs = torch.sigmoid(logits)
            batch_loss = self.dice_weight * soft_dice_loss(probs, masks) + \
                         self.focal_weight * focal_cannab(probs, masks)

            loss.update(batch_loss.item())
            pbar.update(1)
            pbar.set_postfix({'loss': loss()})

        quality_metric = -loss()

        return quality_metric

    def _save_checkpoint(self, checkpoint_path):  
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def train(self, train_data, n_epochs, batch_size, test_data=None, test_batch_size=None,
              last_checkpoint_path=None, best_checkpoint_path=None):

        train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                      shuffle=True, num_workers=self.n_jobs)

        if test_data is not None:
            if test_batch_size is None:
                test_batch_size = batch_size
            test_dataloader = DataLoader(test_data, batch_size=test_batch_size,
                                         shuffle=False, num_workers=self.n_jobs)

        best_metric = float('-inf')
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self._train_epoch(train_dataloader)

            if last_checkpoint_path is not None:
                self._save_checkpoint(last_checkpoint_path)

            if test_data is not None:
                torch.cuda.empty_cache()
                metric = self._test_epoch(test_dataloader)

                if best_checkpoint_path is not None:
                    if metric > best_metric:
                        best_metric = metric
                        self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1