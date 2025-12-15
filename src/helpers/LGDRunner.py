from .BaseRunner import BaseRunner
import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel

class LGDRunner(BaseRunner):
    def __init__(self, args):
        super(LGDRunner, self).__init__(args)
        
    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        dataset.actions_before_epoch()  # must sample before multi thread start

        model.train()
        loss_lst = list()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
						collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)

            # randomly shuffle the items to avoid models remembering the first item being the target
            item_ids = batch['item_id']
            # for each row (sample), get random indices and shuffle the original items
            indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)						
            batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

            model.optimizer.zero_grad()
            out_dict = model(batch)

            # shuffle the predictions back so that the prediction scores match the original order (first item is the target)
            prediction = out_dict['prediction']
            if len(prediction.shape)==2: # only for ranking tasks
                restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
                # use the random indices to shuffle back
                restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction   
                out_dict['prediction'] = restored_prediction

            loss = model.loss(out_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
            
        model.update_count += 1 # 在这里加
        return np.mean(loss_lst).item()