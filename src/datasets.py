import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from sklearn import metrics
from haven import haven_utils as hu
import numpy as np
import tqdm
from torch.utils.data import Subset 
from torch.utils.data import DataLoader

# reference: https://github.com/mlolab/optML-course/blob/main/labs/ex10/ex10.ipynb

def get_dataset(dataset_name, split, datadir, exp_dict):

    train_flag = True if split == 'train' else False
    #  exp_dict["convexity"] in ['stconvex','convex','nonconvex']

    if dataset_name in ['polyhedron_dataset']:
        # A has shape (nb_samples, dim) that stores a_i^T on each row
        # b has shape (nb_samples,)
        # Ax^* <= b
        n = exp_dict["nb_samples"]
        d = exp_dict["d"]
        R = exp_dict['R']

        rng = np.random.default_rng(2024)
        xstar = rng.normal(size=(d), loc = 0) * 100
        xstar = xstar / np.linalg.norm(xstar) * 0.95 * R

        A = rng.random((n,d)) * 2 -1
        Axstar = A @ xstar
        s = np.where(Axstar < 0, Axstar, -np.inf).argmax()
        S = rng.random((n)) * s / 10

        b = A @ xstar + S
        dataset = torch.utils.data.TensorDataset(torch.DoubleTensor(A), torch.DoubleTensor(b))
      

    elif dataset_name == "quadratic":
        n = exp_dict["n_samples"]
        d = exp_dict["d"] 
        convexity = exp_dict["convexity"] 

        rng_A_approx = np.random.default_rng(42)
        rng_noise = np.random.default_rng(43)
        rng_mask = np.random.default_rng(44)
        rng_b = np.random.default_rng(45)

        Avg_approx = (rng_A_approx.random((1,d))*110)
        noise = (rng_noise.random((n,d))*18)

        A = np.clip(Avg_approx + noise, a_min=1, a_max=100)

        if convexity == "convex":
            # A is generated such that the problem is convex
            A[:,:min(20,d)] *= ((2 ** -np.linspace(20,max(1,21-d),21-max(1,21-d))) * n \
                                    / A.sum(0)[:min(20,d)]) 
            mask = rng_mask.choice([True, False], size=(n,min(20,d)), p=[0.5, 0.5])
            mask[0][mask.sum(0)<1] = True
            A[:,:min(20,d)] = A[:,:min(20,d)] * mask
            
        # sample b
        b = (rng_b.random((n,d)) * 10)
        L = A.max()
        print("-------L is {:.2f}-------".format(L))

        dataset = torch.utils.data.TensorDataset(torch.DoubleTensor(A), torch.DoubleTensor(b))
        dataset.targets = b

    if dataset_name == "synthetic_svm":
        margin = exp_dict["margin"]

        X, y, _, _ = make_binary_linear(n=exp_dict["n_samples"],
                                        d=exp_dict["d"],
                                        margin=margin,
                                        y01=True,
                                        bias=True,
                                        separable=exp_dict.get("separable", True),
                                        seed=42)
        # No shuffling to keep the support vectors inside the training set
        splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        X_train, X_test, Y_train, Y_test = splits

        X_train = torch.FloatTensor(X)
        X_test = torch.FloatTensor(X_test)

        Y_train = torch.FloatTensor(y)
        Y_test = torch.FloatTensor(Y_test)

        if train_flag:
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            dataset.targets = Y_train.tolist()
        else:
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)
            dataset.targets = Y_test.tolist()

    if dataset_name in ["mushrooms","a1a","a2a","ijcnn","w8a", \
                        "breast-cancer","duke","leu","phishing","rcv1","epsilon","colon-cancer"]:
        
        X, y = load_libsvm(dataset_name, data_dir=datadir)
       
        labels = np.unique(y)

        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
        # splits used in experiments
        splits = train_test_split(X, y, test_size=0.2, shuffle=True, 
                    random_state=9513451)
        X_train, X_test, Y_train, Y_test = splits


        if train_flag:
            # training set
            X_train = torch.DoubleTensor(X.toarray())
            Y_train = torch.DoubleTensor(y)
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            dataset.targets = Y_train.tolist()
        else:
            # test set
            X_test = torch.DoubleTensor(X_test.toarray())
            Y_test = torch.DoubleTensor(Y_test)
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)
            dataset.targets = Y_test.tolist()

    if dataset_name == "cifar10":
        transform_function_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_function_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=datadir,
            train=train_flag,
            download=False,
            transform=transform_function_train if 
                    train_flag else transform_function_test)

    if dataset_name == "cifar100":
        transform_function_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
       
        transform_function_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function_train if 
                    train_flag else transform_function_test)
        
    if dataset_name == 'svhn':
        
        transform_function = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
            #                     (0.19803012, 0.20101562, 0.19703614))
        ])

        if split == 'train':
            dataset_train = torchvision.datasets.SVHN(
                root=datadir,
                 split='train',
                download=False,
                transform=transform_function)
            
            dataset_extra_train = torchvision.datasets.SVHN(
                root=datadir,
                 split='extra',
                download=False,
                transform=transform_function)
            
            dataset = torch.utils.data.ConcatDataset([
                dataset_train, dataset_extra_train])
            
        else:
            dataset = torchvision.datasets.SVHN(
                root=datadir,
                 split='test',
                download=False,
                transform=transform_function)

        
    return DatasetWrapper(dataset, split=split)

    


class DatasetWrapper:
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return {"images":data, 
                'labels':target, 
                'meta':{'indices':index}}
    

    
# ===========================================================
# Helpers
import os
import urllib

from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from torchvision.datasets import MNIST

def split_index_iid(dataset, nb_users):

    nb_dataset_per_client = int(len(dataset)/nb_users)
    idx_users, all_idxs = {}, [i for i in range(len(dataset))]
    # split the index of whole dataset into nb_users parts
    for i in range(nb_users):
        if i == nb_users - 1:
            size = nb_dataset_per_client + np.mod(len(dataset), nb_users)
        else:
            size = nb_dataset_per_client
        idx_users[i] = np.random.choice(all_idxs, size, replace=False)
        all_idxs = list(set(all_idxs) - set(idx_users[i]))
    return idx_users

def split_index_dirichlet(dataset, nb_users, nb_classes=10, alpha=0.5):

    # generate index for each class
    idx = [torch.where(torch.tensor(dataset.targets) == i)[0].tolist() 
                                                for i in range(nb_classes)]

    # generate the dirichlet distribution with (nb_classes, nb_users)
    # summation of columns = 1
    prob_dist = np.random.dirichlet(np.ones(nb_users) * alpha, nb_classes)

    # fill data_dist with the number of data for each class and each user
    data_dist = np.zeros((nb_classes, nb_users))
    for i in range(nb_classes):
        # multiply probabilities with the number of data for each class
        data_dist[i] = (prob_dist[i] * len(idx[i])).astype('int')
        data_num = data_dist[i].sum()
        # if the number of data is not enough, add the rest of data to a random user 
        data_dist[i][np.random.randint(low=0, high=nb_users)] += (len(idx[i]) - data_num)
        data_dist = data_dist.astype('int')

    # generate index for each client
    idx_users = {}
    for j in range(nb_users):
        idx_users[j] = []
        for i in range(nb_classes):
            d_index = np.random.choice(idx[i], size=data_dist[i][j], replace=False)
            idx_users[j].extend(d_index.tolist())
            idx[i] = list(set(idx[i]) - set(idx_users[j]))

    return idx_users


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"mushrooms"       : "mushrooms",
                      "a1a"             : "a1a",
                      "a2a"             : "a2a",
                      "ijcnn"           : "ijcnn1.tr.bz2",
                      "w8a"             : "w8a",
                      "breast-cancer"   : "breast-cancer",
                      "duke"            : "duke.tr.bz2",
                      "leu"             : "leu.bz2",
                      "phishing"        : "phishing",
                      "rcv1"            : "rcv1_train.binary.bz2",
                      'epsilon'         : "epsilon_normalized.bz2",
                      "colon-cancer"    : "colon-cancer.bz2"}


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y



def make_binary_linear(n, d, margin, y01=False, bias=False, separable=True, scale=1, shuffle=True, seed=None):
    assert margin >= 0.

    if seed:
        np.random.seed(seed)

    labels = [-1, 1]

    # Generate support vectors that are 2 margins away from each other
    # that is also linearly separable by a homogeneous separator
    w = np.random.randn(d); w /= np.linalg.norm(w)
    # Now we have the normal vector of the separating hyperplane, generate
    # a random point on this plane, which should be orthogonal to w
    p = np.random.randn(d-1); l = (-p@w[:d-1])/w[-1]
    p = np.append(p, [l])

    # Now we take p as the starting point and move along the direction of w
    # by m and -m to obtain our support vectors
    v0 = p - margin*w
    v1 = p + margin*w
    yv = np.copy(labels)

    # Start generating points with rejection sampling
    X = []; y = []
    for i in range(n-2):
        s = scale if np.random.random() < 0.05 else 1

        label = np.random.choice(labels)
        # Generate a random point with mean at the center 
        xi = np.random.randn(d)
        xi = (xi / np.linalg.norm(xi))*s

        dist = xi@w
        while dist*label <= margin:
            u = v0-v1 if label == -1 else v1-v0
            u /= np.linalg.norm(u)
            xi = xi + u
            xi = (xi / np.linalg.norm(xi))*s
            dist = xi@w

        X.append(xi)
        y.append(label)

    X = np.array(X).astype(float); y = np.array(y)#.astype(float)

    if shuffle:
        ind = np.random.permutation(n-2)
        X = X[ind]; y = y[ind]

    # Put the support vectors at the beginning
    X = np.r_[np.array([v0, v1]), X]
    y = np.r_[np.array(yv), y]

    if separable:
        # Assert linear separability
        # Since we're supposed to interpolate, we should not regularize.
        clff = SVC(kernel="linear", gamma="auto", tol=1e-10, C=1e10)
        clff.fit(X, y)
        assert clff.score(X, y) == 1.0

        # Assert margin obtained is what we asked for
        w = clff.coef_.flatten()
        sv_margin = np.min(np.abs(clff.decision_function(X)/np.linalg.norm(w)))
        
        if np.abs(sv_margin - margin) >= 1e-4:
            print("Prescribed margin %.4f and actual margin %.4f differ (by %.4f)." % (margin, sv_margin, np.abs(sv_margin - margin)))

    else:
        flip_ind = np.random.choice(n, int(n*0.01))
        y[flip_ind] = -y[flip_ind]

    if y01:
        y[y==-1] = 0

    if bias:
        # TODO: Get rid of this later, bias should be handled internally,
        #       this is just for ease of implementation for the Hessian
        X = np.c_[np.ones(n), X]

    return X, y, w, (v0, v1)


def compute_delta(train_set,idx_users):
    for i in range(len(idx_users)):
        train_subset = Subset(train_set, idx_users[i])
        train_subset_loader = DataLoader(train_subset,
                            drop_last=False,
                            shuffle=True,
                            sampler=None,
                            batch_size=len(train_set))
        pbar = tqdm.tqdm(train_subset_loader, disable=True)
        for batch in pbar:
            A = batch['images'].numpy()
            Avg_i = np.average(A,axis=0)

        if i == 0:
            Avg = Avg_i
        else:
            Avg = np.vstack([Avg,Avg_i])
    
    # compute delta_A and delta_B
    A_Avg = np.average(Avg,axis=0)
    delta_A = np.sqrt( np.average( np.max(np.abs((Avg - A_Avg)), axis=1) ** 2 ) )
    delta_B = np.abs((Avg - A_Avg)).max()
    
    print("-------delta_A is {:.2f}, delta_B is {:.2f}-------".format(delta_A, delta_B))
    return delta_A, delta_B
            