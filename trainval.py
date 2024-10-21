import torch
import pandas as pd
import exp_configs
import time
import os
import numpy as np
from src import models
from src import datasets
from src import utils as ut
from haven import haven_wizard as hw
import argparse
from torch.utils.data.dataloader import default_collate

# cudnn.benchmark = True

'''
Reference: https://github.com/IssamLaradji/sps/blob/master/trainval.py
'''


def trainval(exp_dict, savedir, args):
    # Set seed and device
    # ===================
    seed = 2024 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not available please run with "-c 0"'
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Load Datasets
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split='train',
                                     datadir=args.datadir,
                                     exp_dict=exp_dict)
    # split training dataset
    if exp_dict["split"] == 'iid':
        idx_users = datasets.split_index_iid(train_set.dataset, exp_dict['nb_users'])
    else:
        idx_users = datasets.split_index_dirichlet(train_set.dataset, 
                                    exp_dict['nb_users'], 
                                    nb_classes=exp_dict['nb_classes'], 
                                    alpha=exp_dict['alpha'])

    if exp_dict["dataset"] == 'quadratic':
        delta_A, delta_B = datasets.compute_delta(train_set,idx_users)
   
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                split='val',
                                datadir=args.datadir,
                                exp_dict=exp_dict)

    # Load Model
    # ==================
    model = models.get_model(train_set, exp_dict, idx_users, device=device)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = ut.load_pkl(score_list_path)
        model.set_state_dict(torch.load(model_path))
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0
        
    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["opt"]['max_epoch']):
        # Set seed
        seed = epoch + exp_dict.get('runs', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
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

        # Report and save
        print(pd.DataFrame(score_list).tail())
        ut.save_pkl(score_list_path, score_list)
        # ut.torch_save(model_path, model.get_state_dict())
        print("Saved: %s" % savedir)


   
if __name__ == "__main__":
    import exp_configs
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-c", "--cuda", default=1, type=int)
    parser.add_argument("-j", "--job_scheduler", default=None)
    parser.add_argument("-p", "--python_binary_path", default=None)
    args, others = parser.parse_known_args()

    # Get job configuration to launch experiments in the cluster
    job_config = None
    if os.path.exists('job_configs.py'):
        import job_configs
        job_config = job_configs.JOB_CONFIG

    # Run experiments either sequentially or in the cluster
    hw.run_wizard(func=trainval, 
                exp_groups=exp_configs.EXP_GROUPS, 
                job_config=job_config, 
                job_scheduler=args.job_scheduler,
                python_binary_path=args.python_binary_path,
                use_threads=True, 
                args=args)