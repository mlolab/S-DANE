import numpy as np
import torch
import copy
import tqdm
from torch.utils.data import Subset 
from torch.utils.data import DataLoader

class FedRed(torch.optim.Optimizer):
    '''
    Implements FedRed-SGD / DANE+-SGD / Fedprox / Scaffold / Scaffnew / FedAvg algorithms.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        train_set (TensorDataset): training dataset
        idx_users (dict): index of data splitting for each user
        model_base (base_classifier): classifier model
        loss_function (metric function): loss function
        sigma2 (float): regularization of the loss_function
        lr (float): local learning rate
        step_size_rule (string): ['constant', '1/sqrtk', '1/k']
        outer_lr (float): server-side learning rate (for Scaffold)
        lambda_reguriz (float): lambda defined in the algorithm
        nb_local_steps (int): number of local steps if no_prob is True 
                                    and activate_stopping_criterion is False
        activate_stopping_criterion (bool): activate stopping criterion for DANE+ 
                    or not ||\nabla F_{i,r} (x_{i,r+1})|| \le e^r ||x_{i,r+1} - x^r||
        control_variate (int): type of control variate
        averaging (string): type of averaging strategy ['standard', 'randomized']
        no_prob (bool): no probabilistic averaging or not 
        p (float): averaging probability p
        m (float): the constant used in the second control variate
        batch_size_local_solver (int / string): the batchsize used for local solver
        batch_size_control_variate (int / string): 
                        the batchsize used for the control variate
        nb_clients_sampled (int / string): number of participating clients  
                                if 'full', then all clients are sampled
    '''

    def __init__(self,
                params,
                train_set,
                idx_users,
                model_base,
                loss_function,
                sigma2=0,
                lr=0.1,
                step_size_rule='constant',
                outer_lr=1,
                lambda_reguriz=1e-5,
                nb_local_steps=4,
                activate_stopping_criterion=True,
                control_variate=1,
                averaging='standard',
                no_prob=False,
                p=1/4,
                m=None,
                batch_size_local_solver=32,
                batch_size_control_variate=32,
                nb_clients_sampled='full'              
                ):
        
        if not 0.0 <= sigma2:
            raise ValueError("Invalid regularizer for the loss: {}".format(sigma2))
        if not 0.0 < lr:
            raise ValueError("Invalid local learning rate: {}".format(lr))
        if not (step_size_rule in ['constant', '1/sqrtk', '1/k']):
            raise ValueError("Invalid stepsize update rule: {}".format(step_size_rule))
        if not 0.0 < outer_lr:
            raise ValueError("Invalid server-side learning rate: {}".format(outer_lr))
        if not 0.0 <= lambda_reguriz:
            raise ValueError("Invalid regularizer for the method: {}".format(lambda_reguriz))
        if not 0.0 < nb_local_steps:
            raise ValueError("Invalid number of local steps: {}".format(nb_local_steps))
        if not (isinstance(activate_stopping_criterion, bool)):
            raise ValueError("Invalid choice of activating stopping criterion: {}. Please choose \
                             one bool among [True,False]".format(activate_stopping_criterion))
        if not control_variate in [0,1,2]:
            raise ValueError("Invalid choice of control variate: {}. Please choose \
                             one integer among [0,1,2]".format(control_variate))
        if not averaging in ['standard', 'randomized']:
            raise ValueError("Invalid choice of averaging strategy: {}. Please choose \
                             one string among ['standard', 'randomized']".format(averaging))
        if not (isinstance(no_prob, bool)):
            raise ValueError("Invalid choice of no_prob: {}. \
                             Please choose a bool value".format(no_prob))
        if not 0.0 <= p <= 1:
            raise ValueError("Invalid probability p: {}".format(p))
        if not ((m is None) or (0.0 < m)):
            raise ValueError("Invalid constant : {}".format(m))
        if not (batch_size_local_solver == 'full' or 0.0 < batch_size_local_solver):
            raise ValueError("Invalid batch_size p: {}".format(batch_size_local_solver))
        if not (batch_size_control_variate == 'full' or 0.0 < batch_size_control_variate):
            raise ValueError("Invalid batch_size for control variate p: \
                              {}".format(batch_size_control_variate))
        if not (nb_clients_sampled == 'full' or 0.0 < nb_clients_sampled):
            raise ValueError("Invalid number of clients sampled: \
                              {}".format(nb_clients_sampled))
        
        defaults = dict(
            lr=lr,
            outer_lr=outer_lr,
            lambda_reguriz=lambda_reguriz,
            averaging=averaging,
            no_prob=no_prob,
            step_size_rule=step_size_rule,
            nb_local_steps=nb_local_steps,
            activate_stopping_criterion=activate_stopping_criterion,
            nb_clients_sampled=nb_clients_sampled,
            p=p,
            m=m)
        
        super().__init__(params, defaults)

        self.state['train_set'] = train_set
        self.state['model_base'] = model_base
        self.state['loss_function'] = loss_function
        self.state['idx_users'] = idx_users
        self.state['sigma2'] = sigma2
        self.state['step'] = 0
        self.state['batch_size_local_solver'] = batch_size_local_solver
        self.state['control_variate_type'] = control_variate
        self.state['control_variate'] = {}
        self.state['batch_size_control_variate'] = batch_size_control_variate
        self.state['iterates_list'] = {}
        self.state['total_nb_local_steps'] = 0
        self.state['total_nb_samples'] = compute_nb_samples(idx_users)

        for group in self.param_groups:
            for p in group["params"]:
                device = p.device
                self.state['device'] = device

    def _create_data_loader(self,idx,batch_size):
        # load part of the dataset according to idx
        train_subset = Subset(self.state['train_set'], idx)

        if batch_size == 'full':
            batch_size = len(train_subset)
        else:
            batch_size = batch_size

        train_subset_loader = DataLoader(train_subset,
                            drop_last=False,
                            shuffle=True,
                            sampler=None,
                            batch_size=batch_size)
        
        return train_subset_loader
    
    def compute_full_gradient(self, dataset):
        # compute the full (averaged) gradient for the provided dataset

        device=self.state['device']
        if self.state['batch_size_local_solver'] == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.state['batch_size_local_solver']

        loader = torch.utils.data.DataLoader(
                    dataset, drop_last=False, 
                    shuffle=True,
                    sampler=None,
                    batch_size=batch_size)
        
        pbar = tqdm.tqdm(loader, disable=True)
        
        loss_function = self.state['loss_function'] 
        model_base = self.state['model_base']
        full_grad = []

        for batch in pbar:
            self.zero_grad()
            images, labels = batch["images"].to(device=device), batch["labels"].to(device=device)
            loss = loss_function(model_base, images, labels, sigma2=self.state['sigma2'], backwards=True)
            k=0
            for p in model_base.parameters():
                if len(full_grad) == 0:
                    full_grad = [torch.zeros_like(p.grad,requires_grad=False) 
                                                        for p in model_base.parameters()]
                full_grad[k] += p.grad * images.shape[0] 
                k += 1
 
        for g in full_grad:
            g /= len(dataset)
        
        return full_grad


    def _update_control_variate(self, idx_sampled_clients, 
                                iterates_list=None):
        # update control variate 

        control_variate_list = {}
        grad_list = {}
        if self.state['control_variate_type'] == 0:
            for i in idx_sampled_clients:
                control_variate_list[i] = [torch.zeros_like(p.data,
                                requires_grad=False).to(self.state['device']) 
                                for p in self.state['model_base'].parameters()] 

        # control variate 1: (\nabla f_i - \nabla f)
        # or start of the algorithm for control variate 2
        elif len(self.state['control_variate']) == 0 or \
                self.state['control_variate_type'] == 1:
            
            total_nb_samples = self.state['total_nb_samples']
            nb_clients = len(self.state['idx_users'])
            # create dataset for each user 
            for u in idx_sampled_clients:
                if self.state['batch_size_control_variate'] == 'full':
                    idx = self.state['idx_users'][u]
                else:
                    idx = np.random.choice(self.state['idx_users'][u], 
                        size = self.state['batch_size_control_variate'], replace=False)
                train_subset = Subset(self.state['train_set'], idx)
                # note the true gradient of f_i is 
                # nb_clients * nb_samples_i / total_nb_samples * averaged gradient of client i
                grad_client_i = multiply_number_vector(
                    nb_clients * len(self.state['idx_users'][u]) / total_nb_samples, 
                                        self.compute_full_gradient(train_subset))
                grad_list[u] = grad_client_i
        
            avg_grad = compute_average_vectors(grad_list)

            for u in idx_sampled_clients:
                control_variate_list[u] = add_vectors(1, -1, grad_list[u], avg_grad)

        else:
        # control variate 2: h_{i,r+1} = (m (x^{r+1} - x_{i,r+1}) + h_{i,r})
            for group in self.param_groups:
                params = group["params"]
                lr = group["lr"]
                p = group['p']
                if group['m'] is None:
                    m = p / lr
                else:
                    m = group['m']
            for u in idx_sampled_clients:
                diff = [m*p.data - m*it for p, it in zip(params, iterates_list[u])]
                #diff = add_vectors(m, -m, params, iterates_list[u])
                control_variate_list[u] = \
                    add_vectors(1, 1, diff, self.state['control_variate'][u])
                    
        # update control variate
        self.state['control_variate'] = control_variate_list

    def test_critertion(self, params, idx_user, lambda_reguriz, x_tilde, grad_client, step):
        
        # compute || gradient of F_i ||^2
        norm2_grad_F_i = 0
        for g, c, p, p_s in zip(grad_client, self.state['control_variate'][idx_user],
                                params, x_tilde):
            grad_F_i = g - c + lambda_reguriz * (p.data - p_s.data)
            norm2_grad_F_i += torch.sum(torch.mul(grad_F_i, grad_F_i))
        
        # compute ||x - x_tilde||^2
        norm2_diff = 0
        for p, p_s in zip(params, x_tilde):
            norm2_diff += torch.sum(torch.mul(p.data-p_s.data, p.data-p_s.data))
        
        if norm2_grad_F_i < ((2 * lambda_reguriz ** 2) * norm2_diff / ((step+1)**2) + 1e-6):
            return 'pass'
        else:
            return 'fail'

    def step(self):
        # one communication round of the algorithm

        # get parameters
        for group in self.param_groups:
            
            params = group["params"]
            params_tilde = copy.deepcopy(params)
            lr=group['lr']
            outer_lr=group['outer_lr']
            lambda_reguriz=group['lambda_reguriz']
            step_size_rule=group['step_size_rule']
            activate_stopping_criterion=group['activate_stopping_criterion']
            nb_local_steps=group['nb_local_steps']
            averaging=group['averaging']
            no_prob=group['no_prob']
            nb_clients_sampled=group['nb_clients_sampled']
            p=group['p']
            device = self.state['device']

        total_nb_samples = self.state['total_nb_samples']
        nb_clients = len(self.state['idx_users'])
        if nb_clients_sampled == 'full':
            nb_clients_sampled = nb_clients
        if activate_stopping_criterion:
            nb_local_steps = 0
        
        # get number of local steps
        if (not no_prob):
            nb_local_steps = compute_expected_nb_local_steps(p)

        # sample clients
        idx_sampled_clients = np.random.choice(
                        np.linspace(0,nb_clients-1,nb_clients,dtype=int), 
                        size = nb_clients_sampled, replace=False)

        # update control variate
        self._update_control_variate(idx_sampled_clients,
                                     self.state['iterates_list'])

        # start local iteration for each user
        for i in idx_sampled_clients:

            # reset model parameter to xtilde
            for p, ptilde in zip(params, params_tilde):
                p.data = ptilde.data

            # reset continue flag
            continue_flag = True

            # initialize learning rate
            lr_k = lr

            # record number of local steps
            k = 0

            while (continue_flag and activate_stopping_criterion) \
                or ((k < nb_local_steps) and (not activate_stopping_criterion)):

                # load dataset for user i 
                train_loader_user_i = self._create_data_loader(self.state['idx_users'][i],
                                            self.state['batch_size_local_solver'])
                pbar_i = tqdm.tqdm(train_loader_user_i, disable=True)
                loss_function = self.state['loss_function'] 
           
                for batch in pbar_i:

                    # update stepsize
                    if step_size_rule == 'constant':
                        lr_k = lr 
                    elif step_size_rule == '1/sqrtk':
                        lr_k = lr / (np.sqrt(k + 1))
                    else:
                        lr_k = lr / (k + 1)
                        
                    # copy current model parameter of user i at step k
                    params_step_k = copy.deepcopy(params)
                    self.zero_grad()
                    images, labels = batch["images"].to(device=device), batch["labels"].to(device=device)
                    loss = loss_function(self.state['model_base'], 
                            images, labels, sigma2=self.state['sigma2'], backwards=True)
                    grad_current = get_grad_list(params)
                    # note the true (stochastic) gradient of f_i is 
                    # nb_clients * nb_samples_i / total_nb_samples * averaged (stochastic) gradient of client i
                    grad_current_reweight = multiply_number_vector(
                            nb_clients * len(self.state['idx_users'][i]) / total_nb_samples, 
                                                                                grad_current)
                    # compute argmin of F_{i,k}
                    compute_next_iterate(grad_current_reweight, self.state['control_variate'][i], 
                                params_step_k, params_tilde, params, lr_k, lambda_reguriz)
                    k += 1
                    
                    if activate_stopping_criterion:
                        # ||\nabla F_{i,r}(x_{i,r+1})||^2 \le lambda^2 ||\xx_{i,r+1} - vv^r||^2 + epsilon
                        if self.test_critertion(params, i, lambda_reguriz, params_tilde,
                                                 grad_current_reweight, self.state['step']) == 'pass':
                            continue_flag = False
                            nb_local_steps = max(nb_local_steps, k)
                            break
                    else:
                        if k >= nb_local_steps:
                            break

            self.state['iterates_list'][i] = [copy.deepcopy(p.data) for p in params]
        
        # update global parameters
        if averaging == 'standard':
            average_iterates = compute_average_vectors(self.state['iterates_list'])
            for p, p_avg, p_tilde in zip(params, average_iterates, params_tilde):
                p.data = p_tilde.data + outer_lr * (p_avg - p_tilde.data)
        else:
            random_index = np.random.randint(0, len(self.state['iterates_list']))
            for p, p_avg in zip(params, self.state['iterates_list'][random_index]):
                p.data = p_avg

        self.state['step'] += 1

        self.state['total_nb_local_steps'] += nb_local_steps
        
        return self.state['total_nb_local_steps']


# Helper functions
def compute_average_vectors(vector_list):
    # compute average of the vectors
    idx_users = list(vector_list.keys())

    avg_vec = [torch.zeros_like(g, requires_grad=False) 
                                for g in vector_list[idx_users[0]]]
    
    for i in idx_users:
        for j in range(len(avg_vec)):
            avg_vec[j] += vector_list[i][j] 

    for g in avg_vec:
        g /= len(idx_users)
    
    return avg_vec


def add_vectors(a, b, vector1, vector2):
    # a * vector1 + b * vector2
    results = []
    # substract two vectors
    for v1, v2 in zip(vector1, vector2):
        results.append(a * v1 + b * v2)

    return results

def multiply_number_vector(a, vector):
    # a * vector
    results = []

    for v in vector:
        results.append(a * v)

    return results

def compute_expected_nb_local_steps(p):
    k = 1
    while True:
        coin = np.random.binomial(1, p)
        if coin == 1:
            return k 
        else:
            k += 1

def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]
        
def compute_next_iterate(grad_current, 
                     control_variate, 
                     params_current, 
                     params_tilde,
                     params_next,
                     lr,
                     lambda_reguriz):
    
    for p, p_tilde, p_next, g, h in zip(params_current, 
                params_tilde, params_next, grad_current, control_variate):
        p_next.data = ((h - g + 1/lr*p.data + 
                        lambda_reguriz*p_tilde.data) / (1/lr + lambda_reguriz))
        
    return None

def compute_nb_samples(idx_users):
    nb_samples = 0
    for i in range(len(idx_users)):
        nb_samples += len(idx_users[i])
    return nb_samples