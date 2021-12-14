import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.distributed as dist
import math
import random
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model
from evaluate import mean_average_precision
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from torch.nn import Parameter
def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          multi_labels,
          code_length,
          num_features,
          alpha,
          beta,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_class,
          threshold,
          eta,
          temperature,
          epsilon,
          evaluate_interval,
          ):
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        alpha, beta(float): Hyper-parameters.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.

    Returns
        None
    """
        # generate Hardmard Matrix 
    from scipy.special import comb, perm
    from itertools import combinations
    from scipy.linalg import hadamard


        # hash_targets = torch.nn.Parameter(hash_targets,requires_grad=True)
    
    # Model, optimizer, criterion
    model = load_model(arch, code_length, num_class )
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion_cf = CF_Loss(tau2=2.0)
    criterion = log_cos_Loss()

    features_1, features_2 = extract_features(model, train_dataloader, num_features, device, verbose)
    logger.info('Get the deep feature')
    S_1 = generate_similarity_weight_matrix(features_1, alpha, beta, threshold=threshold, k_positive=400,
                                                 k_negative=2500, Classes=num_class)
    S_1 = S_1.to(device)
    #W_1 = W_1.to(device)
    S_2 = generate_similarity_weight_matrix(features_2, alpha, beta, threshold=threshold, k_positive=400,
                                                 k_negative=2500, Classes=num_class)
    S_2 = S_2.to(device)
    #W_2 = W_2.to(device)
    logger.info('END')


    # Training
    model.train()
    for epoch in range(max_iter):
        n_batch = len(train_dataloader)
        train_loss = AverageMeter()
        cfloss = AverageMeter()
        nceloss = AverageMeter()

        # model.hash_targets.weight = torch.sign(model.hash_targets)
            


        for i, (data, data_aug,_, index) in enumerate(train_dataloader):

            


            data = data.to(device)
            batch_size = data.shape[0]
            data_aug = data_aug.to(device)

            optimizer.zero_grad()

            v= model(data)
            v_aug= model(data_aug)

            #cf loss
            out1 = torch.cat([F.normalize(v.t()), F.normalize(v_aug.t())], dim=0)
            sim_matrix1 = torch.exp(torch.mm(out1, out1.t().contiguous()) / temperature)
            mask1 = (torch.ones_like(sim_matrix1) - torch.eye(2 * code_length, device=sim_matrix1.device)).bool()
            sim_matrix1 = sim_matrix1.masked_select(mask1).view(2 * code_length, -1)
            pos_sim1 = torch.exp(torch.sum(F.normalize(v.t()) * F.normalize(v_aug.t()), dim=-1) / temperature)

            pos_sim1 = torch.cat([pos_sim1, pos_sim1], dim=0)
            cf_loss = (- torch.log(pos_sim1 / sim_matrix1.sum(dim=-1))).mean()

            H = v @ v.t() / code_length
            H_aug = v_aug @ v_aug.t() / code_length
            targets_1 = S_1[index, :][:, index]
            targets_2 = S_2[index, :][:, index]

            loss = cf_loss + (criterion(H, targets_1)+  criterion(H_aug, targets_2))

            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), 1)
            cfloss.update(cf_loss.item(), 1)
            #nceloss.update(nce_loss.item(), 1)


            # Print log
            if verbose:
                logger.info('[iter:{}/{}][loss:{:.4f}]'.format(epoch+1, max_iter, loss.item()))

            # Evaluate
        if (epoch % evaluate_interval == evaluate_interval-1) or (epoch==9) or ((epoch>1000) & (epoch % 100 == 99) ):
            mAP = evaluate(model,
                            query_dataloader,
                            retrieval_dataloader,
                            code_length,
                            device,
                            topk,
                            multi_labels,
                            )
            logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                epoch+1,
                max_iter,
                mAP,
            ))
            logger.info('[iter:{}/{}][total_loss_avg:{:.4f}]'.format(
                epoch + 1,
                max_iter,
                train_loss.avg,
            ))
            logger.info('[iter:{}/{}][cf_loss_avg:{:.4f}]'.format(
                epoch + 1,
                max_iter,
                cfloss.avg,
            ))
            '''
            logger.info('[iter:{}/{}][nce_loss_avg:{:.4f}]'.format(
                epoch + 1,
                max_iter,
                nceloss.avg,
            ))
            '''

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   multi_labels,
                   )
    torch.save({'iteration': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('checkpoints', 'resume_{}_{}.t'.format(code_length, mAP)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, multi_labels):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    model.train()


    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _, index in dataloader:
            data = data.to(device)
            outputs = model(data)
            code[index, :] = outputs.sign().cpu()

    return code



def generate_similarity_weight_matrix(features, alpha, beta, threshold, k_positive, k_negative, Classes):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine'))
    features = features.numpy()
    S = 1-cos_dist*2
 
    return torch.FloatTensor(S)


def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_1 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    features_2 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data_1, data_2 ,_, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            features_1[index, :] = model(data_1).cpu()
            features_2[index, :] = model(data_2).cpu()

    model.set_extract_features(False)
    model.train()
    return features_1, features_2







class CF_Loss(nn.Module):
    def __init__(self, tau2):
        super().__init__()
        self.tau2 = tau2

    def forward(self, ff):

        

        norm_ff = ff / (ff**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(self.tau2)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = F.cross_entropy(coef_mat, a)
        return L_fd

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class log_cos_Loss(nn.Module):
    def __init__(self):
        super(log_cos_Loss, self).__init__()

    def forward(self, H, S):
        loss = ((H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss