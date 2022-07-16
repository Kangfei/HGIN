import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import argparse
import math
from torch.utils.data import DataLoader
from models.predictor import DDGPredictor, E2EGeoPPIDDGPredictor
from utils.data import *
from scripts.train import evaluate
from utils.load import *
from utils.protein import *
from utils.misc import *
from visualize.load import save_pred_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='reg')  # regression ('reg') or 4 class classification ('cla') or Gaussian NLL ('gau)
    parser.add_argument('--res_encoder', type=str, default='mlp')  # mlp or egnn or geoppi
    parser.add_argument('--num_egnn_layers', type=int, default=3,
                        help='number of egnn layers, only used when res_encoder is egnn')
    parser.add_argument('--num_gan_layers', type=int, default=3, help='number of gan layers')
    parser.add_argument('--use_chain_feat', type=bool, default=False, help='whether use chain id for egnn feature input')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--num_neighbors', type=int, default=96, help='number of residues in the PPI interface used')
    #parser.add_argument('--model', type=str, default='/apdcephfs/share_1364275/kfzhao/checkpoints/skempi/ablation/egnn_256/model_400.pt')
    parser.add_argument('--model', type=str, default='/apdcephfs/share_1364275/kfzhao/checkpoints/helixon/gvp_mlp/model_400.pt')
    #parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor2/data/model.pt')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='train & eval batch size') # actual batch size = batch_size * num_gpu * gradient_accumulation
    parser.add_argument('--res_save_path', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor2/result/Helixon/gvpmlp_model_400.csv')
    parser.add_argument('--method', type=str, default='GVPMLP') # method string saved in result file
    #parser.add_argument('--input_data', type=str, default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    #parser.add_argument('--input_data', type=str,  default="/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Evision.pk")
    parser.add_argument('--input_data', type=str, default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi.pk")
    parser.add_argument('--test_input_data', type=str, default="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    seed_all(args.seed)


    ckpt = torch.load(args.model)
    config = ckpt['config']
    config.model['mode'] = args.mode
    config.model["res_encoder"] = args.res_encoder
    config.model["use_chain_feat"] = args.use_chain_feat
    config.model["num_egnn_layers"] = args.num_egnn_layers
    config.model["geomattn"]["num_layers"] = args.num_gan_layers

    print(config)
    if args.res_encoder == "egnn" or args.res_encoder == "mlp":
        print("use ddg predictor model")
        model = DDGPredictor(config.model).to(args.device)
    else:
        print("use GeoPPI model")
        model = E2EGeoPPIDDGPredictor(config.model).to(args.device)
    # load model parameters
    weight = ckpt['model']
    #print(weight.keys())
    model.load_state_dict(weight)

    """
    ## para model average
    model_checkpoints = ['/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/ckpt/egnn_0520/model_490.pt',
               ]
    all_models = list()
    for model_checkpoint in model_checkpoints:
        ckpt = torch.load(model_checkpoint)
        config = ckpt['config']
        config.model['mode'] = args.mode
        config.model["res_encoder"] = args.res_encoder
        config.model["use_chain_feat"] = args.use_chain_feat
        config.model["num_egnn_layers"] = args.num_egnn_layers

        print(config)

        model = DDGPredictor(config.model).to(args.device)
        # load model parameters
        weight = ckpt['model']
        model.load_state_dict(weight)
        all_models.append(model)
    model = model_average_ensemble(all_models)
    """

    #res, mut_info = load_P36_5D2_ground_truth(args.neu_path, args.mut_pdb_dir, args.wt_pdb)
    #res, mut_info = load_input_pk(args.input_data)

    res = load_input_pk(args.input_data)
    print(len(res))
    random.shuffle(res)

    # train_data, val_data = res, None
    print(int(0.8 * len(res)))
    # train_data, val_data = res[1265: ], res[: 1265]
    # train_data, val_data = res[: int(0.8 * len(res))], res[int(0.8 * len(res)):]
    #train_data, val_data, test_data = res[: int(0.6 * len(res))], res[int(0.6 * len(res)): int(0.8 * len(res))], res[int(0.8 * len(res)):]

    test_data, _ = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk")
    #test_data, mut_info = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Independent_bbk.pk")

    test_dataset = DDGDataset(test_data, args.num_neighbors, args.mode)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    criterion = get_criterion_function(args.mode)
    eval_loss, correct, all_preds, all_targets = evaluate(args, model, val_loader=test_loader, criterion=criterion)
    save_pred_result(preds=all_preds, targets=all_targets, output_path=args.res_save_path, method=args.method)
    """
    all_preds, all_targets = all_preds.tolist(), all_targets.tolist()
    for mut, pred, target in zip(mut_info, all_preds, all_targets):
        print(mut, pred, target)
    """

"""
    pred_save_path = "/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/pred_mlp.pk"
    with open(pred_save_path, "wb") as out_file:
        pred_res = {"preds": all_preds, "targets": all_targets}
        pickle.dump(pred_res, out_file)
        out_file.close()
        print("save pred res file in {}".format(pred_save_path))
"""



"""
# for debug only
if __name__ == '__main__':
    with open("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi_bbk.pk", 'rb') as in_file:
        res = pickle.load(in_file)
        in_file.close()
        for (data_wt, data_mut, ddG) in res:
            transform = PPIResidue()
            collate_fn = PaddingCollate()
            mutation_mask = (data_wt['aa'] != data_mut['aa'])  # flag indicate the difference of residual type # [L]

            batch = collate_fn([transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask})])
            #print(batch)
"""