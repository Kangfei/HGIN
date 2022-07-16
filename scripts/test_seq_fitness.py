import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
import random


from models.sequence.rcnn import PIPR
from models.sequence.bert import BertPIPR

from utils.misc import *
from utils.load import load_input_pk
from utils.data import *
from utils.protein import *
from scripts.train import evaluate
from parallel.trainer import *
import logging
from easydict import EasyDict as edict
from embe.embedding import BertLMTokenizer, Word2VecTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--mode', type=str, default='reg') # regression ('reg') or 4 class classification ('cla') or Gaussian NLL ('gau)
    parser.add_argument('--hid_channels', type=int, default=128, help="number of hidden channels in the sequence model")
    parser.add_argument('--use_bert', type=bool, default=False, help='whether or not use bert to generate sequence embedding')
    parser.add_argument('--use_chain_feat', type=bool, default=False, help='whether egnn feature input use chain id')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--batch_size', type=int, default=2, help='test batch size')
    parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/ckpt/rrm/PIPR/model_140.pt')
    #parser.add_argument('--input_data', type=str, default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi.pk")
    parser.add_argument('--input_data', type=str,  default="/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Evision.pk")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    print(args)

    config = None
    seed_all(args.seed)

    if args.use_bert:
        model = BertPIPR(hid_channels=args.hid_channels).to(args.device)
        model = load_checkpoint(model, model_checkpoint=args.model)
        embed_tokenizer = BertLMTokenizer()
    else:
        model = PIPR(in_channels=7, hid_channels=args.hid_channels).to(args.device)
        model = load_checkpoint(model, model_checkpoint=args.model)
        embed_tokenizer = Word2VecTokenizer()

    res_single = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_single.pk")
    res_double = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_double.pk")
    # test_data, mut_info = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    res = res_single + res_double
    print(len(res))
    random.shuffle(res)

    # train_data, val_data = res, None
    print(int(0.8 * len(res)))
    train_data, val_data, test_data = res[: int(0.6 * len(res))], res[int(0.6 * len(res)): int(0.8 * len(res))], res[int(0.8 * len(res)):]
    test_dataset = FitnessDataset(test_data, mode=args.mode, tokenizer=embed_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    criterion = get_criterion_function(args.mode)
    eval_loss, correct, all_preds, all_targets = evaluate(args, model, val_loader=test_dataloader, criterion=criterion)
    print(eval_loss, correct, all_preds.shape, all_targets.shape)
    print("Spearman coef={:.4f}, Person coef = {:4f}".format(spearman(all_preds, all_targets),
                                                             pearson(all_preds, all_targets)))