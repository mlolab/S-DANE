import numpy as np
from torch.utils.data import Subset 
from torch.utils.data import DataLoader
import tqdm
import torch

class GD(torch.optim.Optimizer):
    '''
    Implements GD algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        train_set (TensorDataset): training dataset
        idx_users (dict): index of data splitting for each user
        model_base (base_classifier): classifier model
        loss_function (metric function): loss function
        sigma2 (float): regularization of the loss_function
        lr (float): learning rate
    '''

    def __init__(self,
                params,
                train_set,
                idx_users,
                model_base,
                loss_function,
                sigma2=0,
                lr=0.01):
        
        defaults = dict(lr=lr)
        
        super().__init__(params, defaults)

        self.state['train_set'] = train_set
        self.state['model_base'] = model_base
        self.state['loss_function'] = loss_function
        self.state['idx_users'] = idx_users
        self.state['sigma2'] = sigma2
        self.state['step'] = 0
        self.state['total_nb_local_steps'] = 0

        for group in self.param_groups:
            for p in group["params"]:
                device = p.device
                self.state['device'] = device

    def step(self):
        # one communication round of the algorithm

        # get parameters
        for group in self.param_groups:
            params = group["params"]
            lr=group['lr']

        loader = DataLoader(
                self.state['train_set'], drop_last=False, 
                shuffle=True,
                sampler=None,
                batch_size=len(self.state['train_set']))
        pbar = tqdm.tqdm(loader)
        device = self.state['device']

        loss_function = self.state['loss_function']
        for batch in pbar:
            self.zero_grad()
            images, labels = batch["images"].to(device=device), batch["labels"].to(device=device)
            loss = loss_function(self.state['model_base'], 
                        images, labels, sigma2=self.state['sigma2'], backwards=False)
            loss.backward()

            for p in params:
                p.data -= lr * p.grad

        self.state['total_nb_local_steps'] += 1

        return self.state['total_nb_local_steps']

        