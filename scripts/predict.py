import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch

from models.predictor import DDGPredictor
from utils.misc import *
from utils.data import *
from utils.protein import *


def predict_one(model, wt_pdb_file, mut_pdb_file):
    batch = load_wt_mut_pdb_pair(wt_pdb_file, mut_pdb_file)
    batch = recursive_to(batch, args.device)
    with torch.no_grad():
        model.eval()
        pred, wt_alpha, mut_alpha = model(batch)
        print('Predicted ddG: %.4f' % pred.item())
    wt_alpha = wt_alpha.squeeze().detach().cpu().numpy()
    mut_alpha = mut_alpha.squeeze().detach().cpu().numpy()
    print(wt_alpha.shape, mut_alpha.shape)
    save_path = "/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor2/result/Helixon/alpha_Ag72F_Ng76L_Rg122M_KB417T_EB484K_NB501Y.npz"
    np.savez(save_path, wt_alpha =wt_alpha, mut_alpha =mut_alpha)
    return pred.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='reg')  # regression ('reg'), to do gau and cla!!!
    parser.add_argument('--res_encoder', type=str, default='egnn')  # mlp or egnn
    parser.add_argument('--num_egnn_layers', type=int, default=3,
                        help='number of egnn layers, only used when res_encoder is egnn')
    parser.add_argument('--use_chain_feat', type=bool, default=False, help='whether egnn feature input use chain id')
    #parser.add_argument('--wt_pdb', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/TSINGHUA_data/PDBs/tsinghua_camel_rcsb.pdb')
    #parser.add_argument('--mut_pdb', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/TSINGHUA_data/tsinghua_camel-GB100Y_LB31R_SB29R.pdb')
    parser.add_argument('--wt_pdb', type=str,
                        default='/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/PDBs/7faf_rcsb.pdb')
    parser.add_argument('--mut_pdb', type=str,
                        default='/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Mutants/7faf/7faf-Qg124F_NB439K.pdb')
    parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor2/ckpt/egnn_0520/model_400.pt')
    #parser.add_argument('--model', type=str, default='./data/model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    batch = load_wt_mut_pdb_pair(args.wt_pdb, args.mut_pdb)


    ckpt = torch.load(args.model)
    config = ckpt['config']
    config.model['mode'] = args.mode
    config.model["res_encoder"] = args.res_encoder
    config.model["use_chain_feat"] = args.use_chain_feat
    config.model["num_egnn_layers"] = args.num_egnn_layers
    weight = ckpt['model']
    model = DDGPredictor(config.model).to(args.device)
    model.load_state_dict(weight)

    ddG = predict_one(model, args.wt_pdb, args.mut_pdb)