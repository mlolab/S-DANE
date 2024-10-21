import numpy as np
import torch
import copy
import tqdm
from torch.utils.data import Subset 
from torch.utils.data import DataLoader

class SDANE(torch.optim.Optimizer):
    '''
    Implements Stabalized DANE and its accelerated version with local (S)GD.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        train_set (TensorDataset): training dataset
        idx_users (dict): index of data splitting for each user
        model_base (base_classifier): classifier model
        loss_function (metric function): loss function
        sigma2 (float): regularization of the loss_function
        mu (float): strong convexity parameter
        a (float): the stepsize for updating v if non-accelerated version is used
                by default, a = 1 / lambda_reguriz
        acceleration (bool): activate accelerated version of the algorithm or not
        lr (float): local learning rate
        step_size_rule (string): ['constant', '1/sqrtk', '1/k']
        lambda_reguriz (float): lambda defined in the algorithm
        adaptive_lambda (bool): whether to use adaptive line search or not
        lower_bound_lambda (bool or float or None): whether to use lower bound for lambda or not
                        if False, then no lower bound is used
                        if float, then the lower bound is the provided value
                        if None, then the lower bound is lambda
        record_local_similarity (bool): record local similarity or not
        record_local_smoothness (bool): record local smoothness or not
        nb_local_steps (int): number of local steps if stopping criterion is not used
        activate_stopping_criterion (bool): activate stopping criterion or not
            ||\nabla F_{i,r} (x_{i,r+1})|| \le (\lambda_reguriz / 2) ||x_{i,r+1} - (y or v)^r||
        use_control_variate (bool): incoroporate control variate or not
        epoch_start_control_variate (int): epoch to start using control variate
        batch_size_local_solver (int / string): the batchsize used for local solver
                                if 'full', then the full dataset is used
        batch_size_control_variate (int / string) the batchsize used for computation of 
                                full gradient, if 'full', then the full dataset is used
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
                mu=0,
                a=None,
                acceleration=False,
                lr=0.1,
                step_size_rule='constant',
                lambda_reguriz=1e-5,
                adaptive_lambda=True,
                lower_bound_lambda=False,
                record_local_similarity=False,
                record_local_smoothness=False,
                nb_local_steps=4,
                activate_stopping_criterion=True,
                use_control_variate=True,
                epoch_start_control_variate=0,
                batch_size_local_solver=32,
                batch_size_control_variate='full',
                nb_clients_sampled='full'           
                ):
        
        if not 0.0 <= sigma2:
            raise ValueError("Invalid regularizer for the loss: {}".format(sigma2))
        if not 0.0 <= mu:
            raise ValueError("Invalid strong convexity parameter for the loss: {}".format(mu))
        if not 0.0 < lr:
            raise ValueError("Invalid local learning rate: {}".format(lr))
        if not (step_size_rule in ['constant', '1/sqrtk', '1/k']):
            raise ValueError("Invalid stepsize update rule: {}".format(step_size_rule))
        if not (isinstance(acceleration, bool)):
            raise ValueError("Invalid choice of acceleration: {}".format(acceleration))
        if not 0.0 <= lambda_reguriz:
            raise ValueError("Invalid regularizer for the method: {}".format(lambda_reguriz))
        if not 0.0 < nb_local_steps:
            raise ValueError("Invalid number of local steps: {}".format(nb_local_steps))
        if not (isinstance(activate_stopping_criterion, bool)):
            raise ValueError("Invalid choice of activating stopping criterion: {}. Please choose \
                             one bool among [True,False]".format(activate_stopping_criterion))
        if not (isinstance(adaptive_lambda, bool)):
            raise ValueError("Invalid choice of using adaptive lambda: {}. Please choose \
                             one bool among [True, False]".format(adaptive_lambda))
        if not (isinstance(record_local_similarity, bool)):
            raise ValueError("Invalid choice of recording local similarity: {}. Please choose \
                             one bool among [True, False]".format(record_local_similarity))
        if not (isinstance(record_local_smoothness, bool)):
            raise ValueError("Invalid choice of recording local smoothness: {}. Please choose \
                                one bool among [True, False]".format(record_local_smoothness))
        if not (isinstance(use_control_variate, bool)):
            raise ValueError("Invalid choice of using control variate: {}. Please choose \
                             one bool among [True, False]".format(use_control_variate))
        if not (batch_size_local_solver == 'full' or 0.0 < batch_size_local_solver):
            raise ValueError("Invalid batch size for local SGD: {}".format(batch_size_local_solver))
        if not (batch_size_control_variate == 'full' or 0.0 < batch_size_control_variate):
            raise ValueError("Invalid batch size used in the control \
                                    variate: {}".format(batch_size_control_variate))
        if not (nb_clients_sampled == 'full' or 0.0 < nb_clients_sampled):
            raise ValueError("Invalid number of clients sampled: \
                              {}".format(nb_clients_sampled))
        
        defaults = dict(
            lr=lr,
            mu=mu,
            a=a,
            step_size_rule=step_size_rule,
            acceleration=acceleration,
            adaptive_lambda=adaptive_lambda,
            lower_bound_lambda=lower_bound_lambda,
            record_local_similarity=record_local_similarity,
            record_local_smoothness=record_local_smoothness,
            nb_local_steps=nb_local_steps,
            activate_stopping_criterion=activate_stopping_criterion,
            nb_clients_sampled=nb_clients_sampled,
            epoch_start_control_variate=epoch_start_control_variate
            )
        
        super().__init__(params, defaults)

        self.state['train_set'] = train_set
        self.state['model_base'] = model_base
        self.state['loss_function'] = loss_function
        self.state['idx_users'] = idx_users
        self.state['sigma2'] = sigma2
        self.state['step'] = 0
        self.state['control_variate'] = {}
        self.state['use_control_variate'] = use_control_variate
        self.state['x_iterates_list'] = {}
        self.state['x_grads_list'] = {}
        self.state['total_nb_local_steps'] = 0
        self.state['batch_size_local_solver'] = batch_size_local_solver
        self.state['batch_size_control_variate'] = batch_size_control_variate
        self.state['total_nb_samples'] = compute_nb_samples(idx_users)
        self.state['lambda'] = lambda_reguriz
        if type(lower_bound_lambda) == float:
            self.state['lambda_min'] = lower_bound_lambda
        else:
            self.state['lambda_min'] = lambda_reguriz
        self.state['A'] = 0
        self.state['B'] = 1
        if record_local_similarity:
            self.state['local_similarity'] = None
        if record_local_smoothness:
            self.state['local_smoothness'] = None

        for group in self.param_groups:
            for p in group["params"]:
                device = p.device
                self.state['device'] = device
            self.state['v'] = copy.deepcopy(group["params"])
            if record_local_similarity or record_local_smoothness or adaptive_lambda:
                self.state['old_y_or_v'] = copy.deepcopy(self.state['v'])
    
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
    
    def compute_control_variates(self, idx_sampled_clients):
        # compute control variates at the current params
        grad_list = {}
        control_variate_list = {}
        total_nb_samples = self.state['total_nb_samples']
        nb_clients = len(self.state['idx_users'])
        # create dataset for each sampled user i 
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

        return avg_grad, control_variate_list

    def _update_control_variate(self, idx_sampled_clients, record_local_similarity, 
                                record_local_smoothness, norm2_diff_y_v=None):
        
        # compute control variates
        if record_local_similarity or record_local_smoothness or \
                            self.state['use_control_variate']:
            
            avg_grad, control_variate_list = self.compute_control_variates(idx_sampled_clients) 

            # record local smoothness and local similarity
            if record_local_similarity and self.state['step'] > 0:
                norm2_diff_h = 0
                for i in range(len(self.state['idx_users'])):
                    for h_old, h_new in zip(self.state['control_variate'][i], control_variate_list[i]):
                        norm2_diff_h += torch.sum(torch.mul(h_old-h_new, h_old-h_new))
                norm2_diff_h /= len(self.state['idx_users'])
                self.state['local_similarity'] = torch.sqrt(norm2_diff_h / norm2_diff_y_v).item()

            if record_local_smoothness and self.state['step'] > 0:
                for avg_g_old, avg_g_new in zip(self.state['old_avg_grad'], avg_grad):
                    norm2_diff_g = torch.sum(torch.mul(avg_g_old-avg_g_new, avg_g_old-avg_g_new))
                self.state['local_smoothness'] = torch.sqrt(norm2_diff_g / norm2_diff_y_v).item()
                
            if record_local_smoothness:
                self.state['old_avg_grad'] = avg_grad
        
            self.state['control_variate'] = control_variate_list

    def update_a(self, lambda_reguriz):
        # lambda = (A + a) * B / (a ** 2)
        A = self.state['A']
        B = self.state['B']
        a = (B / (2 * lambda_reguriz)) + np.sqrt(
                            ((B**2) / (4 * lambda_reguriz**2))
                             + (A*B / (lambda_reguriz)))
        return a
    
    def test_critertion(self, params, idx_user, lambda_reguriz, y_or_v, grad_client):
        
        # compute || gradient of F_i ||^2
        norm2_grad_F_i = 0
        for g, c, p, p_s in zip(grad_client, self.state['control_variate'][idx_user],
                                params, y_or_v):
            grad_F_i = g - c + lambda_reguriz * (p.data - p_s.data)
            norm2_grad_F_i += torch.sum(torch.mul(grad_F_i, grad_F_i))
        
        # compute ||x - y or v||^2
        norm2_diff = 0
        for p, p_s in zip(params, y_or_v):
            norm2_diff += torch.sum(torch.mul(p.data-p_s.data, p.data-p_s.data))
        
        if norm2_grad_F_i < ((lambda_reguriz ** 2 / 4) * norm2_diff):
            return 'pass'
        else:
            return 'fail'
        
    def check_adaptive_acceptance(self, idx_sampled_clients, y_or_v,
                                  x_average_iterates, lambda_reguriz):
        
        norm2_avg_grad = 0
        grad_average = compute_average_vectors(self.state['x_grads_list'])
        for g, g in zip(grad_average, grad_average):
            norm2_avg_grad += torch.sum(torch.mul(g, g)) 

        avg_inner_product = 0
        # compute control variates at x_average_iterates
        for group in self.param_groups:
            params = group["params"]
        for p, p_avg in zip(params, x_average_iterates):
            p.data = p_avg.data
        _, control_variate_list = self.compute_control_variates(idx_sampled_clients) 
        for i in idx_sampled_clients:
            for g, h, y, x in zip(self.state['x_grads_list'][i], control_variate_list[i], 
                                  y_or_v, self.state['x_iterates_list'][i]):
                avg_inner_product += torch.sum(torch.mul(g - h, y - x))
        avg_inner_product /= len(idx_sampled_clients)
        if avg_inner_product >= (norm2_avg_grad / (2 * lambda_reguriz)):
            return 'accept'
        else:    
            return 'reject'


    def step(self):
        # one communication round of the algorithm

        # get parameters
        for group in self.param_groups:
            
            params = group["params"]
            lr=group['lr']
            mu=group['mu']
            a=group['a']
            step_size_rule=group['step_size_rule']
            device = self.state['device']
            acceleration=group['acceleration']
            adaptive_lambda=group['adaptive_lambda']
            lower_bound_lambda=group['lower_bound_lambda']
            record_local_similarity=group['record_local_similarity']
            record_local_smoothness=group['record_local_smoothness']
            nb_local_steps=group['nb_local_steps']
            activate_stopping_criterion=group['activate_stopping_criterion']
            nb_clients_sampled=group['nb_clients_sampled']
            epoch_start_control_variate=group['epoch_start_control_variate']

        total_nb_samples = self.state['total_nb_samples']
        nb_clients = len(self.state['idx_users'])
        if nb_clients_sampled == 'full':
            nb_clients_sampled = nb_clients
        if activate_stopping_criterion:
            nb_local_steps = 0

        if nb_clients_sampled < nb_clients:
            # only support full-client participation for now
            adaptive_lambda = False
            record_local_similarity = False
            record_local_smoothness = False

            # sample clients
            idx_sampled_clients = np.random.choice(
                        np.linspace(0,nb_clients-1,nb_clients,dtype=int), 
                        size = nb_clients_sampled, replace=False)
        else:
            idx_sampled_clients = np.linspace(0,nb_clients-1,
                                              nb_clients,dtype=int)

        lambda_reguriz = self.state['lambda']
        if adaptive_lambda and self.state['step'] > 0:
            if self.state['adaptive_accept']:
                if lower_bound_lambda is False:
                    lambda_reguriz /= 2
                else:
                    lambda_reguriz = max(lambda_reguriz / 2, 
                                     self.state['lambda_min'])
            else:
                lambda_reguriz *= 2
        
        if adaptive_lambda:
            params_old = copy.deepcopy(params)

        # y or v is stored in params
        if acceleration:
            A = self.state['A']
            B = self.state['B']
            if a is None:
                a = self.update_a(lambda_reguriz)
            linear_combination(params,A/(A+a),params,a/(A+a),self.state['v'])
        else:
            if a is None:
                a = 1 / lambda_reguriz
            linear_combination(params,0,params,1,self.state['v'])

        if ((self.state['step'] > 0) and (record_local_similarity is True 
                                          or record_local_smoothness is True)):
            norm2_diff_y_v = 0
            for p, p_s in zip(params, self.state['old_y_or_v']):
                norm2_diff_y_v += torch.sum(torch.mul(p.data-p_s.data, p.data-p_s.data))
        
        else:
            norm2_diff_y_v = None

        y_or_v = copy.deepcopy(params)

        if (self.state['step'] == 0) and (self.state['use_control_variate'] is False):
            self.state['use_control_variate_flag'] = False
        elif (self.state['step'] == 0) and (self.state['use_control_variate'] is True):
            self.state['use_control_variate_flag'] = True

        if (self.state['step'] >= epoch_start_control_variate
            ) and (self.state['use_control_variate_flag'] is True):
            self.state['use_control_variate'] = True
        else:
            self.state['use_control_variate'] = False

        # update control variate at y or v 
        # update local similarity and local smoothness if required
        if not (self.state['step'] > 0 and adaptive_lambda and 
                (not self.state['adaptive_accept'])):
            self._update_control_variate(idx_sampled_clients, 
                    record_local_similarity, record_local_smoothness, norm2_diff_y_v)

        continue_flag = True

        # start local computation for each user
        for i in idx_sampled_clients:

            # reset model parameter to y or v
            for p, p_s in zip(params, y_or_v):
                p.data = p_s.data

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
                    # compute one (S)GD step 
                    if self.state['use_control_variate']:
                        compute_next_iterate(grad_current_reweight, self.state['control_variate'][i], 
                                    params_step_k, y_or_v, params, lr_k, lambda_reguriz)
                    else:
                        compute_next_iterate(grad_current_reweight, 
                                             [torch.zeros_like(p.data, 
                                                requires_grad=False).to(device=device) for p in params], 
                                            params_step_k, y_or_v, params, lr_k, lambda_reguriz)
                    k += 1
                    
                    if activate_stopping_criterion:
                        # ||\nabla F_{i,r}(x_{i,r+1})||^2 \le lambda^2 / 4 * ||\xx_{i,r+1} - vv^r||^2 
                        if self.test_critertion(params, i, lambda_reguriz, y_or_v, grad_current_reweight) == 'pass':
                            continue_flag = False
                            nb_local_steps = max(nb_local_steps, k)
                            break
                    else:
                        if k >= nb_local_steps:
                            break
            
            # update x_iterates_list
            self.state['x_iterates_list'][i] = [copy.deepcopy(p.data) for p in params]
            # update x_grads_list
            subset_i = Subset(self.state['train_set'], self.state['idx_users'][i])
            full_grad_client_i = multiply_number_vector(
                nb_clients * len(self.state['idx_users'][i]) / total_nb_samples, 
                                self.compute_full_gradient(subset_i))
            self.state['x_grads_list'][i] = full_grad_client_i

        x_average_iterates = compute_average_vectors(self.state['x_iterates_list'])
        x_grad_average = compute_average_vectors(self.state['x_grads_list'])

        # check adaptive acceptance condition
        if adaptive_lambda:
            accept_result = self.check_adaptive_acceptance(
                                idx_sampled_clients, y_or_v,
                                x_average_iterates, lambda_reguriz)
            
        if adaptive_lambda and accept_result == 'reject':
        
            self.state['adaptive_accept'] = False
            # reset parameters to the previous step
            for p, p_old in zip(params, params_old):
                p.data = p_old.data
        else:
            
            # update global parameters x and v
            for p, p_avg in zip(params, x_average_iterates):
                p.data = p_avg.data 
            
            for v_next, v, x, g in zip(self.state['v'], self.state['v'], 
                                      x_average_iterates, x_grad_average):
                if acceleration:
                    v_next.data = (mu * a) / (mu * a + B) * x + B / (mu * a + B) \
                                * v - a / (mu * a + B) * g
                else:
                    v_next.data = (mu * a) / (mu * a + 1) * x + 1 / (mu * a + 1) \
                                * v - a / (mu * a + 1) * g
                    
            if adaptive_lambda and accept_result == 'accept':
                self.state['adaptive_accept'] = True

            # update A and B
            if acceleration:
                self.state['A'] += a 
                self.state['B'] += (mu * a)

            # record y or v 
            if record_local_similarity or record_local_smoothness or adaptive_lambda:
                self.state['old_y_or_v'] = copy.deepcopy(y_or_v)
                
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