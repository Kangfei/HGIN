import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
import random
from torch.utils.data import DataLoader


from models.predictor_pipool import DDGPredictorPiPool
from utils.misc import *
from utils.load import *
from utils.data import *
from utils.protein import *
from parallel.trainer_multitask import MultiTaskParallelTrainer
from parallel.scheduler import NoamLR
import time


def evaluate(args, model, val_loader, criterion):
    model.eval()
    eval_loss = 0.0
    total_correct = 0
    all_preds = list()
    all_targets = list()
    start = time.time()
    for step, batch in enumerate(val_loader):
        with torch.no_grad():
            batch = recursive_to(batch, device=args.device)
            pred, _ = model(batch)
            if args.mode == 'cla':
                loss = criterion(pred, batch['ddG'])
                pred = torch.nn.Softmax(dim=1)(pred)
                pred = torch.argmax(pred, dim=1)
                sign = (pred - 1.5) * (batch['ddG'] - 1.5)
            elif args.mode == 'reg':
                loss = criterion(pred, batch['ddG'].float())
                sign = pred * batch['ddG'].float()
            else:
                mean, var = pred[:, 0], pred[:, 1]
                var = torch.nn.functional.softplus(var) + 1e-6
                loss = criterion(mean, batch['ddG'], var)
                sign = mean * batch['ddG'].float()
            correct = torch.where(sign > 0, torch.ones_like(sign), torch.zeros_like(sign)).sum(-1)
            eval_loss += loss.item()
            total_correct += correct.item()
            all_preds.append(mean.detach().cpu()) if args.mode == 'gau' else all_preds.append(pred.detach().cpu())
            all_targets.append(batch['ddG'].float().detach().cpu())
    end = time.time()
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_preds, all_targets = all_preds.numpy().ravel(), all_targets.numpy().ravel()
    mse = (np.square(all_preds - all_targets)).mean(axis=-1)
    print("Total test time: {:.4f} seconds, eval loss= {:.4f}, mse= {:.4f}, Spearman={:.4f}, Person = {:4f}, correct={}"
          .format((end - start), eval_loss, mse, spearman(all_preds, all_targets), pearson(all_preds, all_targets), total_correct))
    return eval_loss, total_correct, all_preds, all_targets



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training hyper-parameters
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--warmup_steps", default=200, type=int,
                        help="number of warmup steps for Noam scheduler")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_factor', type=float, default=0.5,
                        help='decay rate of (gamma).')
    parser.add_argument('--decay_patience', type=int, default=10,
                        help='num of epochs for one lr decay.')
    parser.add_argument('--min_lr', type=float, default=1e-06,
                        help='minimum lr')
    parser.add_argument('--lr_scheduler_type', type=str, default="Plateau",
                        help='lr scheduler type: Noam or Plateau')
    parser.add_argument('--max_grad_norm', type=float, default=50.0,
                        help='max gradient norm for gradient clip')
    parser.add_argument('--batch_size', type=int, default=1,
                        # actual batch size = batch_size * num_gpu * gradient_accumulation
                        help='train & eval batch size')
    parser.add_argument("--gradient_accumulation",type=int,default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--alpha", type=float, default=0.5, help="loss coefficient of multitask", )
    # model hyper-parameters
    parser.add_argument('--mode', type=str, default='reg') # regression ('reg') or 4 class classification ('cla') or Gaussian NLL ('gau)
    parser.add_argument('--res_encoder', type=str, default='egnn')  # mlp or egnn or geoppi
    parser.add_argument('--use_chain_feat', type=bool, default=False, help='whether egnn feature input use chain id')
    parser.add_argument('--num_egnn_layers', type=int, default=3, help='number of egnn layers, only used when res_encoder is egnn')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num_neighbors', type=int, default=48, help='number of residues in the PPI interface used')
    parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor2/data/model.pt')
    parser.add_argument('--ckpt_freq', type=int, default=10, help='frequency of model checkpoint')
    parser.add_argument('--save_ckpt_dir', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor2/ckpt/egnn_pipool')
    parser.add_argument('--input_data', type=str,  default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi_ppicode_bbk.pk")
    parser.add_argument('--wandb_log_dir', type=str, default='/apdcephfs/share_1364275/kfzhao/wandb_log')
    parser.add_argument('--enable_wandb', type=bool, default=True)
    parser.add_argument('--test_input_data', type=str, default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
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

    model = DDGPredictorPiPool(config.model).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.lr_scheduler_type == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.decay_factor, patience=args.decay_patience, min_lr=args.min_lr)
    else:
        scheduler = NoamLR(optimizer=optimizer, warmup_steps=args.warmup_steps)
    res = load_input_pk(args.input_data)
    print(len(res))
    random.shuffle(res)

    #train_data, val_data = res, None
    print(int(0.8 * len(res)))
    #train_data, val_data = res[1265: ], res[: 1265]
    train_data, val_data = res[: int(0.8 * len(res))], res[int(0.8 * len(res)):]
    #train_data, val_data, test_data = res[: int(0.6 * len(res))], res[int(0.6 * len(res)): int(0.8 * len(res))], res[int(0.8 * len(res)):]
    test_data, mut_info = load_input_pk(args.test_input_data)
    train_dataset = DDGDatasetWithPiPool(train_data, args.num_neighbors, args.mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PaddingCollate())

    val_dataset = DDGDatasetWithPiPool(val_data, args.num_neighbors, args.mode)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    test_dataset = DDGDatasetWithPiPool(test_data, args.num_neighbors, args.mode)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=PaddingCollate())

    criterion = get_criterion_function(args.mode)
    trainer = MultiTaskParallelTrainer(args, config, model, criterion, optimizer, scheduler)
    trainer.train(train_loader=train_dataloader, eval_loader=val_dataloader, test_loader=test_dataloader)

