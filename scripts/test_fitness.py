import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import argparse
import math
from torch.utils.data import DataLoader
from models.predictor import FitnessPredictor, E2EGeoPPIFitnessPredictor
from utils.data import *
from scripts.train import evaluate
from utils.load import *
from utils.protein import *
from utils.misc import *
from visualize.load import save_pred_result
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='reg')  # regression ('reg') or 4 class classification ('cla') or Gaussian NLL ('gau)
    parser.add_argument('--res_encoder', type=str, default='egnn')  # mlp or egnn
    parser.add_argument('--num_egnn_layers', type=int, default=3,
                        help='number of egnn layers, only used when res_encoder is egnn')
    parser.add_argument('--use_chain_feat', type=bool, default=False, help='whether use chain id for egnn feature input')
    parser.add_argument('--num_neighbors', type=int, default=96, help='number of residues in the PPI interface used')
    parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/ckpt/rrm/egnn/model_0.pt')
    #parser.add_argument('--model', type=str, default='./data/model.pt')
    parser.add_argument('--batch_size', type=int, default=1,
                        # actual batch size = batch_size * num_gpu * gradient_accumulation
                        help='train & eval batch size')
    parser.add_argument('--neu_path', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/P36-5D2/neutralization.xlsx')
    parser.add_argument('--mut_pdb_dir', type=str,  default='/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/P36-5D2/mut_pdbs')
    parser.add_argument('--wt_pdb', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/P36-5D2/P36-5D2_7FAF_FvAg.pdb')
    parser.add_argument('--input_data', type=str,
                        default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()


    ckpt = torch.load(args.model)
    config = ckpt['config']
    config.model['mode'] = args.mode
    config.model["res_encoder"] = args.res_encoder
    config.model["use_chain_feat"] = args.use_chain_feat
    config.model["num_egnn_layers"] = args.num_egnn_layers

    print(config)

    if args.res_encoder == 'egnn' or args.res_encoder == 'mlp':
        model = FitnessPredictor(config.model).to(args.device)
    else:
        model = E2EGeoPPIFitnessPredictor(config.model).to(args.device)
    # load model parameters
    weight = ckpt['model']
    model.load_state_dict(weight)

    res_single = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_single.pk")
    res_double = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_double.pk")
    # test_data, mut_info = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    res = res_single + res_double
    print(len(res))
    random.shuffle(res)

    # train_data, val_data = res, None
    print(int(0.8 * len(res)))
    _, _, test_data = res[: int(0.6 * len(res))], res[int(0.6 * len(res)): int(0.8 * len(res))], res[int(0.8 * len(res)):]

    test_dataset = FitnessDataset(res, args.mode)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    criterion = get_criterion_function(args.mode)
    eval_loss, correct, all_preds, all_targets = evaluate(args, model, val_loader=test_loader, criterion=criterion)

    """
    all_preds, all_targets = all_preds.tolist(), all_targets.tolist()
    for pred, target in zip(all_preds, all_targets):
        print(pred, target)
    """

