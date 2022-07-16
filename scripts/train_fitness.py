import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
import random
from torch.utils.data import DataLoader


from models.predictor import FitnessPredictor, E2EGeoPPIFitnessPredictor
from utils.misc import *
from utils.load import *
from utils.data import *
from utils.protein import *
from parallel.trainer import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training hyper-parameters
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_factor', type=float, default=0.5,
                        help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10,
                        help='num of epochs for one lr decay.')
    parser.add_argument('--min_lr', type=float, default=1e-06,
                        help='minimum lr')
    parser.add_argument('--max_grad_norm', type=float, default=50.0,
                        help='max gradient norm for gradient clip')
    parser.add_argument('--batch_size', type=int, default=1,
                        # actual batch size = batch_size * num_gpu * gradient_accumulation
                        help='train & eval batch size')
    parser.add_argument("--gradient_accumulation",type=int,default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # model hyper-parameters
    parser.add_argument('--mode', type=str, default='reg') # regression ('reg') or 4 class classification ('cla') or Gaussian NLL ('gau)
    parser.add_argument('--res_encoder', type=str, default='egnn')  # mlp or egnn or geoppi
    parser.add_argument('--use_chain_feat', type=bool, default=False, help='whether egnn feature input use chain id')
    parser.add_argument('--num_egnn_layers', type=int, default=3, help='number of egnn layers, only used when res_encoder is egnn')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/data/model.pt')
    parser.add_argument('--ckpt_freq', type=int, default=10, help='frequency of model checkpoint')
    parser.add_argument('--save_ckpt_dir', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/ckpt/rrm_double/egnn')
    parser.add_argument('--input_data', type=str, default="/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_single.pk")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    seed_all(args.seed)



    ckpt = torch.load(args.model)
    config = ckpt['config']
    config.model["mode"] = args.mode
    config.model["res_encoder"] = args.res_encoder
    config.model["use_chain_feat"] = args.use_chain_feat
    config.model["num_egnn_layers"] = args.num_egnn_layers

    print(config)

    if args.res_encoder == 'egnn' or args.res_encoder == 'mlp':
        model = FitnessPredictor(config.model).to(args.device)
    else:
        model = E2EGeoPPIFitnessPredictor(config.model).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience, min_lr=args.min_lr)
    #res_single = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_single.pk")
    #res_double = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_double.pk")
    # test_data, mut_info = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    res = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Envision_fitness.pk")
    print(len(res))
    random.shuffle(res)

    # train_data, val_data = res, None
    print(int(0.8 * len(res)))
    train_data, val_data, test_data = res[: int(0.6 * len(res))], res[int(0.6 * len(res)): int(0.8 * len(res))], res[int(0.8 * len(res)):]

    train_dataset = FitnessDataset(train_data, args.mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PaddingCollate())

    val_dataset = FitnessDataset(val_data, args.mode)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    test_dataset = FitnessDataset(test_data, args.mode)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    criterion = get_criterion_function(args.mode)
    trainer = ParallelTrainer(args, config, model, criterion, optimizer, scheduler)
    trainer.train(train_loader=train_dataloader, eval_loader=val_dataloader, test_loader=test_dataloader)
