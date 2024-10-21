import torch 
import tqdm
from torch.utils.data import DataLoader 

def get_metric_function(metric_name):

    if metric_name == "quadratic_loss":
        return quadratic_loss
    
    elif metric_name == "quadratic_acc":
        return quadratic_acc
    
    elif metric_name == "logistic_loss":
        return logistic_loss
    
    elif metric_name == "logistic_acc":
        return logistic_accuracy
    
    elif metric_name == "softmax_accuracy":
        return softmax_accuracy

    elif metric_name == "softmax_loss":
        return softmax_loss
    
    elif metric_name == "polyhedron_loss":
        return polyhedron_loss
    

def polyhedron_loss(model, images, labels, sigma2=None, backwards=False):
    # return Avg((<a_i, x> - b_i)_+^q)
    # each row of A is a_i
    A = model(images)
    for p in model.parameters():
        loss = (torch.nn.functional.relu(A @ p.view(-1) - labels) ** 2).mean()
    if backwards and loss.requires_grad:
        loss.backward()
    return loss 

def quadratic_loss(model, images, labels, sigma2=None, backwards=True):
    # return .5*(w-b)^TA(w-b) + sigma2* \sum_{i} { w[i] / (w[i] + 1) }
    A = model(images)
    for p in model.parameters():
        loss = ((p - labels) * A * (p - labels)).sum()/labels.shape[0]

    if sigma2 is not None:
        w = 0.
        for p in model.parameters():
            w += torch.sum((p**2) / (1 + p**2))
            
        loss += sigma2 * w

    if backwards and loss.requires_grad:
        loss.backward()
   
    return loss

def quadratic_acc(model, images, labels, sigma2=None, backwards=False):
    # return .5*(w-b)^TA(w-b) - f^\star 
    # only implemented for sigma^2 = None, otherwise the metric is gradient norm
    A = model(images)
    for p in model.parameters():
        loss = ((p - labels) * A * (p - labels)).sum()/labels.shape[0]
        xstar = 1 / images.sum(0) * (images * labels).sum(0)
        fstar = ((xstar - labels) * images * (xstar - labels)).sum()/labels.shape[0]

    if backwards and loss.requires_grad:
        loss.backward()
    
    return loss - fstar + 1e-40

def logistic_loss(model, images, labels, sigma2=None, backwards=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.view(-1))

    if sigma2 is not None:
        w = 0.
        for p in model.parameters():
            w += torch.sum(p**2)

        loss += sigma2 / 2 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_accuracy(model, images, labels, sigma2=None):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_loss(model, images, labels, sigma2=None, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if sigma2 is not None:
        w = 0.
        for p in model.parameters():
            w += torch.sum(p**2)

        loss += sigma2 / 2 * w

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def softmax_accuracy(model, images, labels, sigma2=None):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc
