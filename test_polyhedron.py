import torch
import pandas as pd
import time
import numpy as np
from src import models
from src import datasets


def test_polyhedron():
    # Set seed and device
    # ===================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'

    print("\n------Test of S-DANE running on a polyhedron feasibility problem "+
          "with n = 1000, d = 100, R = 5.------")

    print('Running on device: %s' % device)

    opt_test = {'name':'S-DANE', 
                'max_epoch': 50, 
                'acceleration': False,
                'lr': 0.8, 
                'mu': 0.1,
                'step_size_rule': 'constant',
                'lambda_reguriz': 0.1, 
                'adaptive_lambda': False,
                'activate_stopping_criterion': True,
                'use_control_variate': True,
                'nb_clients_sampled': 10,
                'batch_size_local_solver': 'full', 
                'batch_size_control_variate': 'full'}
    
    exp_dict = {"dataset":'polyhedron_dataset',
                    "nb_samples": 1000,
                    "d": 100,
                    "model":'polyhedron_linear_model',
                    "loss_func": 'polyhedron_loss',
                    "acc_func": 'polyhedron_loss',
                    "opt": opt_test,
                    'R': 5,
                    'sigma2': None,
                    "record_grad_norm": False,
                    "split": 'iid',
                    'batch_size_val_acc': 200,
                    'nb_users': 10,
                    "runs": 0}

    # Load Datasets
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split='train',
                                     datadir='',
                                     exp_dict=exp_dict)
    # split training dataset
    if exp_dict["split"] == 'iid':
        idx_users = datasets.split_index_iid(train_set.dataset, exp_dict['nb_users'])
    else:
        idx_users = datasets.split_index_dirichlet(train_set.dataset, 
                                    exp_dict['nb_users'], 
                                    nb_classes=exp_dict['nb_classes'], 
                                    alpha=exp_dict['alpha'])
   
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                split='val',
                                datadir='',
                                exp_dict=exp_dict)

    # Load Model
    # ==================
    model = models.get_model(train_set, exp_dict, idx_users, device=device)
    score_list = []
    s_epoch = 0
        
    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["opt"]['max_epoch']):

        # Record metrics
        score_dict = {"epoch": epoch}
        
        # Validate one epoch
        train_loss_dict, grad_norm_dict = model.val_on_trainset(
                            train_set, metric=exp_dict["loss_func"], name='loss',
                            record_grad_norm=exp_dict["record_grad_norm"])
        val_acc_dict = model.val_on_testset(
                            val_set, metric=exp_dict["acc_func"], name='acc')
        
        score_dict.update(train_loss_dict)
        score_dict.update(val_acc_dict)
        if exp_dict["record_grad_norm"]:
            score_dict.update(grad_norm_dict)
        if model.opt.state.get("A") is not None:
            score_dict["A"] = model.opt.state.get("A")
        if model.opt.state.get("lambda") is not None:
            score_dict["lambda"] = model.opt.state.get("lambda")    
        if model.opt.state.get("local_similarity") is not None:
            score_dict["local_similarity"] = model.opt.state.get("local_similarity")
        if model.opt.state.get("local_smoothness") is not None:
            score_dict["local_smoothness"] = model.opt.state.get("local_smoothness")

        # Train one epoch (or equivalent one communication round)
        s_time = time.time()
        total_nb_local_steps = model.train_one_epoch()
        e_time = time.time()

        if epoch == 0:
            score_dict['total_nb_local_steps'] = 0
        else:
            score_dict['total_nb_local_steps'] = total_nb_local_steps
        score_dict["train_epoch_time"] = e_time - s_time

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report
        print(pd.DataFrame(score_list).tail())

    assert train_loss_dict['train_loss'] < 1e-8

   
if __name__ == "__main__":
    test_polyhedron()