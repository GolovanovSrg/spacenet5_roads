import os
import random

import torch
import torch.nn as nn
import apex

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from optim import RAdam
from loss import soft_dice_loss, focal_cannab
from utils import multichannel_jaccard


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
        min_lr = optimizer_params.get('min_lr', 1e-8)
        lr_factor = optimizer_params.get('lr_factor', 0.1)
        lr_patience = optimizer_params.get('lr_patience', 10)
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
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    
    def _train_epoch(self, train_dataloader):
        pbar = tqdm(desc=f'Train, epoch #{self.last_epoch}', total=len(train_dataloader))
        sampler_weights = train_dataloader.sampler.weights.clone()
        self.model.train()
        
        loss, focal_loss, dice_loss = AvgMeter(), AvgMeter(), AvgMeter()
        for images, masks, idxs in train_dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            probs = torch.sigmoid(logits)

            batch_focal_loss = focal_cannab(probs, masks)
            batch_dice_loss = soft_dice_loss(probs, masks)
            batch_loss = self.focal_weight * batch_focal_loss + \
                         self.dice_weight * batch_dice_loss

            smooth_factor = 0.2
            for k, idx in enumerate(idxs):
                sampler_weights[idx] = (1 - smooth_factor) * sampler_weights[idx] + smooth_factor * batch_loss.data[k]

            batch_focal_loss = batch_focal_loss.mean()
            batch_dice_loss = batch_dice_loss.mean()
            batch_loss = batch_loss.mean()

            self.optimizer.zero_grad()

            with apex.amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                    
            self.optimizer.step()

            loss.update(batch_loss.item())
            focal_loss.update(batch_focal_loss.item())
            dice_loss.update(batch_dice_loss.item())

            pbar.update(1)
            pbar.set_postfix({'loss': loss(),
                              'focal_loss': focal_loss(),
                              'dice_loss': dice_loss()})
        
        train_dataloader.sampler.weights[:] = sampler_weights[:]

    @torch.no_grad()
    def _test_epoch(self, test_dataloader):
        pbar = tqdm(desc=f'Test, epoch #{self.last_epoch}', total=len(test_dataloader))
        self.model.eval()

        loss, focal_loss, dice_loss = AvgMeter(), AvgMeter(), AvgMeter()
        jaccard = AvgMeter()
        for images, masks, _ in test_dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5)

            batch_focal_loss = focal_cannab(probs, masks).mean()
            batch_dice_loss = soft_dice_loss(probs, masks).mean()
            batch_loss = self.focal_weight * batch_focal_loss + \
                         self.dice_weight * batch_dice_loss
            batch_jaccard = multichannel_jaccard(preds, masks)

            loss.update(batch_loss.item())
            focal_loss.update(batch_focal_loss.item())
            dice_loss.update(batch_dice_loss.item())
            jaccard.update(batch_jaccard.item())

            pbar.update(1)
            pbar.set_postfix({'loss': loss(),
                              'focal_loss': focal_loss(),
                              'dice_loss': dice_loss(),
                              'jaccard': jaccard()})

        quality_metric = -loss()

        return quality_metric

    def _save_checkpoint(self, checkpoint_path):  
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def train(self, train_data, n_epochs, batch_size, test_data=None, test_batch_size=None,
              last_checkpoint_path=None, best_checkpoint_path=None):

        train_sampler = WeightedRandomSampler([1] * len(train_data), len(train_data))
        train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                      sampler=train_sampler, num_workers=self.n_jobs)

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
                self.scheduler.step(metric)
                
                if best_checkpoint_path is not None:
                    if metric > best_metric:
                        best_metric = metric
                        self._save_checkpoint(best_checkpoint_path)

            self.last_epoch += 1