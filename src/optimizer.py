from . import optimizers
import torch 
import numpy as np

def get_optimizer(opt_dict, params, train_set, idx_users, model_base, loss_function, sigma2):

    opt_name = opt_dict['name']
        
    if any(['FedRed' in opt_name, 'Scaffnew' in opt_name, 
            'Scaffold' in opt_name, 'Fedprox' in opt_name, 
            'DANE+' in opt_name, 'FedAvg' in opt_name]):
    
        opt = optimizers.FedRed(params, 
                        train_set=train_set,
                        idx_users=idx_users,
                        model_base=model_base,
                        loss_function=loss_function,
                        sigma2=sigma2,
                        lr=opt_dict.get("lr", .1),
                        outer_lr=opt_dict.get("outer_lr", 1),
                        step_size_rule=opt_dict.get("step_size_rule", 'constant'),
                        lambda_reguriz=opt_dict.get("lambda_reguriz", .1),
                        nb_local_steps=opt_dict.get("nb_local_steps", 4),
                        activate_stopping_criterion=opt_dict.get("activate_stopping_criterion", False),
                        control_variate=opt_dict.get("control_variate", 0),
                        averaging=opt_dict.get("averaging", 'standard'),
                        no_prob=opt_dict.get("no_prob", False),
                        p=opt_dict.get("p", 1/4),
                        m=opt_dict.get("m", None),
                        batch_size_local_solver=opt_dict.get("batch_size_local_solver", 'full'),
                        batch_size_control_variate=opt_dict.get(
                            "batch_size_control_variate", 'full'),
                        nb_clients_sampled=opt_dict.get("nb_clients_sampled", 'full')
                        )
        
    elif any(['S-DANE' in opt_name, 'Acc-S-DANE' in opt_name]):

        opt = optimizers.SDANE(params,
                        train_set=train_set,
                        idx_users=idx_users,
                        model_base=model_base,
                        loss_function=loss_function,
                        sigma2=sigma2,
                        mu=opt_dict.get('mu', 0),
                        a=opt_dict.get('a', None),
                        acceleration=opt_dict.get("acceleration", False),
                        lr=opt_dict.get("lr", .1),
                        step_size_rule=opt_dict.get("step_size_rule", 'constant'),
                        lambda_reguriz=opt_dict.get("lambda_reguriz", .1),
                        record_local_similarity=opt_dict.get("record_local_similarity", False),
                        record_local_smoothness=opt_dict.get("record_local_smoothness", False),
                        adaptive_lambda=opt_dict.get("adaptive_lambda", False),
                        lower_bound_lambda=opt_dict.get("lower_bound_lambda", False),
                        nb_local_steps=opt_dict.get("nb_local_steps", 4),
                        activate_stopping_criterion=opt_dict.get("activate_stopping_criterion", False),
                        use_control_variate=opt_dict.get("use_control_variate", True),
                        epoch_start_control_variate=opt_dict.get("epoch_start_control_variate", 0),
                        batch_size_local_solver=opt_dict.get("batch_size_local_solver", 'full'),
                        batch_size_control_variate=opt_dict.get("batch_size_control_variate", 'full'),
                        nb_clients_sampled=opt_dict.get("nb_clients_sampled", 'full'))
        
    elif any(['AccGradSliding' in opt_name]):
        opt = optimizers.AccGradSliding(params,
                        train_set=train_set,
                        idx_users=idx_users,
                        model_base=model_base,
                        loss_function=loss_function,
                        sigma2=sigma2,
                        mu=opt_dict.get('mu', 0),
                        lr=opt_dict.get("lr", .1),
                        step_size_rule=opt_dict.get("step_size_rule", 'constant'),
                        lambda_reguriz=opt_dict.get("lambda_reguriz", .1),
                        adaptive_lambda=opt_dict.get("adaptive_lambda", False),
                        nb_local_steps=opt_dict.get("nb_local_steps", 4),
                        activate_stopping_criterion=opt_dict.get("activate_stopping_criterion", False),
                        batch_size_local_solver=opt_dict.get("batch_size_local_solver", 'full'),
                        batch_size_control_variate=opt_dict.get("batch_size_control_variate", 'full'),
                        idx_comp=opt_dict.get("idx_comp",0))
                
    elif opt_name in ['GD']:
        opt = optimizers.GD(params, 
                        train_set=train_set,
                        idx_users=idx_users,
                        model_base=model_base,
                        loss_function=loss_function,
                        sigma2=sigma2,
                        lr=opt_dict.get("lr",0.01))
    # others        
    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt





