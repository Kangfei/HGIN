import random
import torch
import torch.linalg
import numpy as np
from easydict import EasyDict as edict
from scipy import stats

#Original configuation in PNAS'22
origin_config = edict({'model': {'node_feat_dim': 128, 'pair_feat_dim': 64, 'max_relpos': 32, 'geomattn': {'num_layers': 3, 'spatial_attn_mode': 'CB'}},
          'train': {'loss_weights': {'ddG': 1.0}, 'max_iters': 10000000, 'val_freq': 1000, 'batch_size': 8, 'seed': 2021, 'max_grad_norm': 50.0,
                    'optimizer': {'type': 'adam', 'lr': 0.0001, 'weight_decay': 0.0, 'beta1': 0.9, 'beta2': 0.999},
                    'scheduler': {'type': 'plateau', 'factor': 0.5, 'patience': 10, 'min_lr': 1e-06}},
          'datasets': {'train': {'dataset_path': './data/skempi.pt'}, 'val': {'dataset_path': './data/skempi.pt'}}})


def load_checkpoint(model, model_checkpoint: str):
    ckpt = torch.load(model_checkpoint)
    weight = ckpt['model']
    model.load_state_dict(weight)
    return model

def save_checkpoint(config, model, model_checkpoint:str):
    ckpt = {'config': config, "model": model.state_dict()}
    torch.save(ckpt, model_checkpoint)
    print("save DDGPredictor in {}!".format(model_checkpoint))


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def get_criterion_function(mode: str = 'reg'):
    if mode == 'cla':
        return torch.nn.CrossEntropyLoss()
    elif mode == 'reg':
        return torch.nn.MSELoss()
    elif mode == 'gau':
        return torch.nn.GaussianNLLLoss()
    else:
        raise NotImplementedError("Loss mode: {} is not supported!".format(mode))


def model_average_ensemble(all_models: list):
    all_para = list()
    for model in all_models:
        para = model.state_dict()
        all_para.append(para)
    for key in all_para[0]:
        for i in range(1, len(all_models)):
            all_para[0][key] += all_para[i][key]
        all_para[0][key] = all_para[0][key] / float(len(all_models))
    all_models[0].load_state_dict(all_para[0])
    return all_models[0]


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return (recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def pearson(y_pred, y_true):
    y_pred, y_true = y_pred.ravel(), y_true.ravel()
    diff_pred, diff_true = y_pred - np.mean(y_pred), y_true - np.mean(y_true)
    return np.sum(diff_pred * diff_true) / np.sqrt(np.sum(diff_pred ** 2) * np.sum(diff_true ** 2))


def spearman(y_pred, y_true):
    y_pred, y_true = y_pred.ravel(), y_true.ravel()
    return stats.spearmanr(y_pred, y_true).correlation