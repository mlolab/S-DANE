import numpy as np
import torch
import copy
import tqdm
from torch.utils.data import Subset 
from torch.utils.data import DataLoader

class AccGradSliding(torch.optim.Optimizer):
    '''
    Implements Accelerated Extragradient Sliding
    https://arxiv.org/pdf/2205.15136
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        train_set (TensorDataset): training dataset
        idx_users (dict): index of data splitting for each user
        model_base (base_classifier): classifier model
        loss_function (metric function): loss function
        sigma2 (float): regularization of the loss_function
        mu (float): strong convexity parameter
        lr (float): local learning rate
        step_size_rule (string): ['constant', '1/sqrtk', '1/k']
        lambda_reguriz (float): lambda defined in the algorithm
        adaptive_lambda (bool): whether to use adaptive lambda or not
        nb_local_steps (int): number of local steps if stopping criterion is not used
        activate_stopping_criterion (bool): activate stopping criterion or not
            ||\nabla F_{i,r} (x_{i,r+1})|| \le \lambda_reguriz ||x_{i,r+1} - x^r||
        batch_size_local_solver (int / string): the batchsize used for local solver
                                        if 'full', then the full dataset is used
        batch_size_control_variate (int / string) the batchsize used for computation of 
                        full gradient, if 'full', then the full dataset is used
        idx_comp (int) the index of the user that is responsible for doing the local computations.
    '''

    def __init__(self,
                params,
                train_set,
                idx_users,
                model_base,
                loss_function,
                sigma2=0,
                mu=0,
                lr=0.1,
                step_size_rule='constant',
                lambda_reguriz=1e-5,
                adaptive_lambda=False,
                nb_local_steps=4,
                activate_stopping_criterion=True,
                batch_size_local_solver=32,
                batch_size_control_variate='full',
                idx_comp=0         
                ):
        
        if not 0.0 <= sigma2:
            raise ValueError("Invalid regularizer for the loss: {}".format(sigma2))
        if not 0.0 <= mu:
            raise ValueError("Invalid strong convexity parameter for the loss: {}".format(mu))
        if not 0.0 < lr:
            raise ValueError("Invalid local learning rate: {}".format(lr))
        if not (step_size_rule in ['constant', '1/sqrtk', '1/k']):
            raise ValueError("Invalid stepsize update rule: {}".format(step_size_rule))
        if not 0.0 <= lambda_reguriz:
            raise ValueError("Invalid regularizer for the method: {}".format(lambda_reguriz))
        if not (isinstance(activate_stopping_criterion, bool)):
            raise ValueError("Invalid choice of activating stopping criterion: {}. Please choose \
                             one bool among [True,False]".format(activate_stopping_criterion))
        if not 0.0 < nb_local_steps:
            raise ValueError("Invalid number of local steps: {}".format(nb_local_steps))
        if not (batch_size_local_solver == 'full' or 0.0 < batch_size_local_solver):
            raise ValueError("Invalid batch size for local SGD: {}".format(batch_size_local_solver))
        if not (batch_size_control_variate == 'full' or 0.0 < batch_size_control_variate):
            raise ValueError("Invalid batch size used in the control \
                                    variate: {}".format(batch_size_control_variate))
        if not (isinstance(idx_comp, int)):
            raise ValueError("Invalid index of the user that is responsible for \
                             doing the local computations: {}".format(idx_comp))

        defaults = dict(
            lr=lr,
            mu=mu,
            step_size_rule=step_size_rule,
            adaptive_lambda=adaptive_lambda,
            nb_local_steps=nb_local_steps,
            activate_stopping_criterion=activate_stopping_criterion,
            idx_comp=idx_comp
            )
        
        super().__init__(params, defaults)

        self.state['train_set'] = train_set
        self.state['model_base'] = model_base
        self.state['loss_function'] = loss_function
        self.state['idx_users'] = idx_users
        self.state['sigma2'] = sigma2
        self.state['step'] = 0
        self.state['control_variate'] = {}
        self.state['x_iterates_list'] = {}
        self.state['v_iterates_list'] = {}
        self.state['total_nb_local_steps'] = 0
        self.state['batch_size_local_solver'] = batch_size_local_solver
        self.state['batch_size_control_variate'] = batch_size_control_variate
        self.state['total_nb_samples'] = compute_nb_samples(idx_users)
        self.state['lambda_reguriz'] = lambda_reguriz
        self.state['A'] = 0
        self.state['B'] = 1

        for group in self.param_groups:
            for p in group["params"]:
                device = p.device
                self.state['device'] = device
            self.state['v'] = copy.deepcopy(group["params"])
    
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


    def _update_control_variate(self, idx_clients, 
                                adaptive_lambda, 
                                idx_comp=None, norm2_diff_y=None):
        # update control variate 

        control_variate_list = {}
        grad_list = {}

        total_nb_samples = self.state['total_nb_samples']
        nb_clients = len(self.state['idx_users'])
        # create dataset for each sampled user i 
        for u in idx_clients:
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
        
        for u in idx_clients:
            control_variate_list[u] = add_vectors(1, -1, grad_list[u], avg_grad)

        # update control variate
        if adaptive_lambda and self.state['step'] > 0:
            norm2_diff_h = 0
            for h_old, h_new in zip(self.state['control_variate'][idx_comp], 
                                    control_variate_list[idx_comp]):
                norm2_diff_h += torch.sum(torch.mul(h_old-h_new, h_old-h_new))
            new_lambda = torch.sqrt(norm2_diff_h / norm2_diff_y)
            self.state['control_variate'] = control_variate_list
            return new_lambda.item()
        else:
            self.state['control_variate'] = control_variate_list

    def update_a(self, lambda_reguriz):
        # lambda = (A + a) * B / (2 * a ** 2)
        A = self.state['A']
        B = self.state['B']
        a = (B / (4 * lambda_reguriz)) + np.sqrt(
                            ((B**2) / (16 * lambda_reguriz**2))
                             + (A*B / (2*lambda_reguriz)))
        return a
    
    def test_critertion(self, params, idx_user, lambda_reguriz, y, grad_client):
        
        # compute || gradient of F_i ||^2
        norm2_grad_F_i = 0
        for g, c, p, p_s in zip(grad_client, self.state['control_variate'][idx_user],
                                params, y):
            grad_F_i = g - c + lambda_reguriz * (p.data - p_s.data)
            norm2_grad_F_i += torch.sum(torch.mul(grad_F_i, grad_F_i))
        
        # compute ||x - y||^2
        norm2_diff = 0
        for p, p_s in zip(params, y):
            norm2_diff += torch.sum(torch.mul(p.data-p_s.data, p.data-p_s.data))
       
        if norm2_grad_F_i < ((4 * lambda_reguriz ** 2) * norm2_diff + 1e-12):
            return 'pass'
        else:
            return 'fail'

    def step(self):
        # one communication round of the algorithm

        # get parameters
        for group in self.param_groups:
            
            params = group["params"]
            lr=group['lr']
            mu=group['mu']
            step_size_rule=group['step_size_rule']
            adaptive_lambda=group['adaptive_lambda']
            device = self.state['device']
            nb_local_steps=group['nb_local_steps']
            activate_stopping_criterion=group['activate_stopping_criterion']
            idx_comp=group['idx_comp']

        lambda_reguriz = self.state['lambda_reguriz']
        total_nb_samples = self.state['total_nb_samples']
        nb_clients = len(self.state['idx_users'])
        if activate_stopping_criterion:
            nb_local_steps = 0

        # sample clients
        idx_clients = np.linspace(0,nb_clients-1,nb_clients,dtype=int)
        
        # y is stored in params
        A = self.state['A']
        B = self.state['B']
        a = self.update_a(lambda_reguriz)
        linear_combination(params,A/(A+a),params,a/(A+a),self.state['v'])

        if ((self.state['step'] > 0) and (adaptive_lambda is True)):
            norm2_diff_y = 0
            for p, p_s in zip(params, self.state['old_y']):
                norm2_diff_y += torch.sum(torch.mul(p.data-p_s.data, p.data-p_s.data))
     

        y = copy.deepcopy(params)

        if adaptive_lambda is True:
            self.state['old_y'] = copy.deepcopy(y)
            
        # update control variate at y 
        if (self.state['step'] > 0 and adaptive_lambda is True):
            lambda_reguriz = self._update_control_variate(idx_clients, 
                                        adaptive_lambda, idx_comp, norm2_diff_y)
            self.state['lambda_reguriz'] = lambda_reguriz   
        else:
            self._update_control_variate(idx_clients, adaptive_lambda)

        continue_flag = True
      
        # start local computation for user i 
        i = idx_comp
        
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
                # compute one (S)GD step 
                compute_next_iterate(grad_current_reweight, self.state['control_variate'][i], 
                            params_step_k, y, params, lr_k, lambda_reguriz)
                k += 1
                
                if activate_stopping_criterion:
                    # ||\nabla F_{i,r}(x_{i,r+1})||^2 \le lambda^2 ||\xx_{i,r+1} - vv^r||^2 + epsilon
                    if self.test_critertion(params, i, lambda_reguriz, y, grad_current_reweight) == 'pass':
                        continue_flag = False
                        nb_local_steps = max(nb_local_steps, k)
                        break

                else:
                    if k >= nb_local_steps:
                        continue_flag = False
                        break
   
        # update v
        full_grad = self.compute_full_gradient(self.state['train_set'])
        for v_next, p, p_s, g in zip(self.state['v'], params, self.state['v'], full_grad):
            v_next.data = (mu * a) / (mu * a + B) * p + B / (mu * a + B) \
                        * p_s - a / (mu * a + B) * g
        
        # update A and B
        self.state['A'] += a 
        self.state['B'] += (mu * a)
        self.state['lambda'] = lambda_reguriz
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

def linear_combination(y,a,params_1,b,params_2):
    # compute linear combination stored at y
    for y_next, p1, p2 in zip(y, params_1, params_2):
        y_next.data = a * p1.data + b * p2.data

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

def get_grad_list(params):
    return [copy.deepcopy(p.grad) for p in params]
        
def compute_next_iterate(grad_current, 
                     control_variate, 
                     params_current, 
                     params_start_point,
                     params_next,
                     lr,
                     lambda_reguriz):
    
    for p, p_s, p_next, g, h in zip(params_current, 
                params_start_point, params_next, grad_current, control_variate):
        p_next.data = ((h - g + 1/lr*p.data + 
                        lambda_reguriz*p_s.data) / (1/lr + lambda_reguriz))
        
    return None

def compute_nb_samples(idx_users):
    nb_samples = 0
    for i in range(len(idx_users)):
        nb_samples += len(idx_users[i])
    return nb_samples