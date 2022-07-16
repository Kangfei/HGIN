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
        pred = model(batch['wt'], batch['mut'])
        #print('Predicted ddG: %.4f' % pred.item())
    return pred.item()

def run_evoef2_build_one_mutant(wt_pdb_file : str, mut_tags: str, tmp_work_path: str, evoef2_path: str):

    mut_file = os.path.join(tmp_work_path, '{}_mutation.txt'.format(mut_tags))
    #os.makedirs(os.path.dirname(mut_file), exist_ok=True)
    with open(mut_file, 'w') as fp_txt:
        fp_txt.write(mut_tags + ';\n')
    fp_txt.close()
    os.system('{} --command=BuildMutant --pdb={} --mutant_file={}'.format(os.path.join(evoef2_path, 'EvoEF2'), wt_pdb_file, mut_file))
    os.system('rm -rf {}'.format(mut_file))


def predict_one_mutant(model, wt_pdb_file: str, mut_tags: str, tmp_work_path: str, evoef2_path: str):
    run_evoef2_build_one_mutant(wt_pdb_file, mut_tags, tmp_work_path, evoef2_path)
    wt_pdb_name = os.path.splitext(os.path.basename(wt_pdb_file))[0]


    mut_pdb_file = './{}_Model_0001.pdb'.format(wt_pdb_name)
    assert os.path.exists(mut_pdb_file), "mutation pdf file generation failed!"
    ddG = predict_one(model, wt_pdb_file, mut_pdb_file)

    os.system('rm -rf {}'.format(tmp_work_path))
    os.system('rm {}'.format(mut_pdb_file))
    return ddG



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='reg')  # regression ('reg'), to do gau and cla!!!
    parser.add_argument('--res_encoder', type=str, default='egnn')  # mlp or egnn
    parser.add_argument('--num_egnn_layers', type=int, default=3,
                        help='number of egnn layers, only used when res_encoder is egnn')
    parser.add_argument('--wt_pdb', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/P36-5D2/P36-5D2_7FAF_FvAg.pdb')
    parser.add_argument('--evoef2_path', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/EvoEF2',
                        help='the directory of EvoEF2 execution file')
    parser.add_argument('--clean_work_path', type=str, default='/apdcephfs/share_1364275/kfzhao/Bio_data/EvoEF2/tmp',
                        help='tmp working directory to save intermediate file')
    parser.add_argument('--mut_tags', type=str, default='NB501Y')
    parser.add_argument('--model', type=str, default='/apdcephfs/private_coffeezhao/PycharmProjects/DDGPredictor/ckpt/egnn/model_80.pt')
    #parser.add_argument('--model', type=str, default='./data/model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()


    ckpt = torch.load(args.model)
    config = ckpt['config']
    config.model['mode'] = args.mode
    config.model["res_encoder"] = args.res_encoder
    config.model["num_egnn_layers"] = args.num_egnn_layers
    weight = ckpt['model']
    model = DDGPredictor(config.model).to(args.device)
    model.load_state_dict(weight)

    assert os.path.exists(args.wt_pdb), "wildtype pdb file does not exist!"
    assert os.path.exists(os.path.join(args.evoef2_path, 'EvoEF2')), "EvoEF2 executor does not exist!"
    ddG = predict_one_mutant(model, args.wt_pdb, args.mut_tags, args.clean_work_path, args.evoef2_path)
    print("mutation: {}, ddG: {}".format(args.mut_tags, ddG))
