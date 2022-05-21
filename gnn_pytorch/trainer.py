import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader 

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 100
    batch_size = 100
    lr = 0.001
    weight_decay = 0.95
    num_workers = 2
    ckpt_path = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__setattr__(k, v)

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model).to(self.device)
            else:
                self.model = self.model.to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"saving {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else self.model
        optimizer = optim.Adam(
            raw_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        def run_epoch(loader, is_train):
            model.train(is_train)

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, a, y) in pbar:

                x = x.to(self.device)
                a = a.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    pred, loss = model(x, a, y)
                    if torch.cuda.device_count() > 1:
                        loss = loss.mean() # collapse loss in case of multiple gpus
                losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_description(
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}."
                    )
            agg_loss = float(np.mean(losses))
            return agg_loss 

        best_loss = float('inf')
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            self.val_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )

        for epoch in range(config.max_epochs):
            train_loss = run_epoch(train_loader, is_train=True)
            logger.info(f"train_loss: {train_loss:.f}")
            val_loss = run_epoch(val_loader, is_train=False)
            logger.info(f"val_loss: {val_loss:.f}")

            good_model = self.val_dataset is None or val_loss < best_loss
            if config.ckpt_path is not None and good_model:
                best_loss = val_loss
                self.save_checkpoint()
