import torch
from torch import nn
from torch.nn import functional as F
from . import base_classifier
from . import optimizer
import tqdm
from .metrics import get_metric_function

def get_model(train_set, exp_dict, idx_users, device):
    return Classifier(train_set, exp_dict, idx_users, device)

class Classifier(torch.nn.Module):
    def __init__(self, train_set, exp_dict, idx_users, device):
        super().__init__()
        self.exp_dict = exp_dict
        self.device = device
        self.idx_users = idx_users
        
        # Load classifier and loss function
        self.model_base = base_classifier.get_classifier(exp_dict['model'], train_set)
        self.loss_function = get_metric_function(self.exp_dict['loss_func'])

        # Load Optimizer
        self.to(device=self.device)
        sigma2 = self.exp_dict.get('sigma2') if self.exp_dict.get('sigma2') else 0.0
        self.opt = optimizer.get_optimizer(opt_dict=exp_dict["opt"],
                                       params=self.parameters(),
                                       train_set=train_set, 
                                       idx_users = idx_users,                               
                                       model_base=self.model_base,
                                       loss_function=self.loss_function,
                                       sigma2=sigma2)
        
    def val_on_testset(self, dataset, metric, name):

        self.eval()

        sigma2 = self.exp_dict.get('sigma2') if self.exp_dict.get('sigma2') else 0.0
        metric_function = get_metric_function(metric)
        if metric == 'quadratic_acc':
            batch_size = self.exp_dict['n_samples']
        else:
            batch_size=self.exp_dict['batch_size_val_acc']
        loader = torch.utils.data.DataLoader(  
                        dataset, drop_last=False, batch_size=batch_size)

        score_sum = 0.
        pbar = tqdm.tqdm(loader, disable=True)
        for batch in pbar:
            images, labels = batch["images"].to(
                    device=self.device), batch["labels"].to(device=self.device)
            score_sum += metric_function(
                self.model_base, images, labels, sigma2).item() * images.shape[0]    

        score = float(score_sum / len(loader.dataset))
        return {f'{dataset.split}_{name}': score}
    
    def val_on_trainset(self, dataset, metric, record_grad_norm, name):

        sigma2 = self.exp_dict.get('sigma2') if self.exp_dict.get('sigma2') else 0.0
        metric_function = get_metric_function(metric)
        if metric == 'quadratic_acc':
            batch_size = self.exp_dict['n_samples']
        else:
            batch_size=self.exp_dict['batch_size_val_acc']
        loader = torch.utils.data.DataLoader(
                                    dataset, drop_last=False, batch_size=batch_size)
        
        # compute loss
        score_sum = 0.
        pbar = tqdm.tqdm(loader, disable=True)
        grad_store = []
        
        for batch in pbar:
            self.opt.zero_grad()
            images, labels = batch["images"].to(device=self.device), batch["labels"].to(device=self.device)
            score_sum += metric_function(
                self.model_base, images, labels, sigma2, backwards=record_grad_norm).item() * images.shape[0]
            
            if record_grad_norm:
                k=0
                for p in self.model_base.parameters():
                    if len(grad_store) == 0:
                        grad_store = [torch.zeros_like(p.grad,requires_grad=False) 
                                                            for p in self.model_base.parameters()]
                    grad_store[k] += p.grad * images.shape[0] 
                    k += 1
            
        score = float(score_sum / len(dataset))

        # compute grad norm
        if record_grad_norm:
            # compute grad norm
            grad_norm2 = torch.tensor([0]).double()
            for g in grad_store:
                if g is None:
                    continue
                g /= len(dataset)
                grad_norm2 += torch.sum(torch.mul(g, g))
            grad_norm = torch.sqrt(grad_norm2)
        else:
            grad_norm = torch.tensor([-1])

        return {f'{dataset.split}_{name}': score}, {'grad_norm': grad_norm.item()}

    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def set_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        self.opt.load_state_dict(state_dict["opt"])

    
    def train_one_epoch(self):

        self.train()
        total_nb_local_steps = self.opt.step()

        return total_nb_local_steps



