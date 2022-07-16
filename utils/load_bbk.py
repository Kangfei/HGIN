import pandas as pd
import os
import math
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.protein import parse_pdb
from multiprocessing import Pool
import pickle
import re

def get_ppi_code(data_wt, p1_chain_ids, p2_chain_ids):
    ppi_code = []
    for chain_id in data_wt['chain_id']:
        if chain_id in p1_chain_ids:
            ppi_code.append(0)
        elif chain_id in p2_chain_ids:
            ppi_code.append(1)
        else:
            print(p1_chain_ids, p2_chain_ids, chain_id)
            assert False, "Cannot recognize the chain_id: {}".format(chain_id)
    ppi_code = torch.LongTensor(ppi_code)
    return ppi_code

def process_row(row):
    ddG = 0 # row['delta_delta_g']

    pdb_id = row['pdb_code']
    mutations = '_'.join(row['mutations'].split(';')[0].split(','))
    mut_pdb_file = "{}-{}.pdb".format(pdb_id, mutations)
    #wt_pdb_file = "{}.pdb".format(pdb_id)
    wt_pdb_file = "{}_truncate.pdb".format(pdb_id)
    print(mut_pdb_file, wt_pdb_file)
    #wt_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/SKEMPI2/PDBs", wt_pdb_file)
    #mut_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/SKEMPI2/Mutants", pdb_id, mut_pdb_file)
    wt_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Independent/PDBs", wt_pdb_file)
    mut_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Independent/Mutants", pdb_id, mut_pdb_file)
    #wt_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Helixon/PDBs",  wt_pdb_file)
    #mut_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Helixon/Mutants", pdb_id, mut_pdb_file)
    #wt_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/TSINGHUA/PDBs", wt_pdb_file)
    #mut_pdb_path = os.path.join("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/TSINGHUA/Mutants", pdb_id, mut_pdb_file)
    print(wt_pdb_path)
    print(mut_pdb_path)


    if pd.isna(row['antibody_light_chain_id']):
        p1_chain_ids, p2_chain_ids = row['antibody_heavy_chain_id'], row['antigen_chain_id']
    else:
        p1_chain_ids, p2_chain_ids = row['antibody_heavy_chain_id'] + row['antibody_light_chain_id'], row['antigen_chain_id']
    use_chain_ids = p1_chain_ids + p2_chain_ids

    print(os.path.exists(mut_pdb_path), os.path.exists(wt_pdb_path))
    #data_mut = parse_pdb(mut_pdb_path)
    #data_wt = parse_pdb(wt_pdb_path)

    data_mut = parse_pdb(mut_pdb_path, use_chain_ids=use_chain_ids)
    data_wt = parse_pdb(wt_pdb_path, use_chain_ids=use_chain_ids)

    ppi_code = get_ppi_code(data_wt, p1_chain_ids, p2_chain_ids)
    assert ppi_code.shape[0] == data_wt['aa'].shape[0], "inconsistent chain length"
    data_wt['ppi_code'] = ppi_code
    data_mut['ppi_code'] = ppi_code

    print(data_wt['aa'].shape[0], data_mut['aa'].shape[0], ddG)
    return data_wt, data_mut, ddG, mutations

def process_rosetta_row(row):

    ddG = row['delta_delta_g']
    pdb_id = row['pdb_code']
    mutations = '_'.join(row['mutations'].split(';')[0].split(','))
    #pdb_path = os.path.join('/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Helixon/RosettaMutants/7faf/', mutations)
    pdb_path = os.path.join('/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/SKEMPI2/RosettaMutants/', pdb_id, mutations)
    print(pdb_path)

    if pd.isna(row['antibody_light_chain_id']):
        p1_chain_ids, p2_chain_ids = row['antibody_heavy_chain_id'], row['antigen_chain_id']
    else:
        p1_chain_ids, p2_chain_ids = row['antibody_heavy_chain_id'] + row['antibody_light_chain_id'], row['antigen_chain_id']
    use_chain_ids = p1_chain_ids + p2_chain_ids

    for pdb_file in os.listdir(pdb_path):

        if re.match(r'MUT(.*)bj3.pdb', pdb_file):
            mut_pdb_path = os.path.join(pdb_path, pdb_file)
            print(mut_pdb_path)
            data_mut = parse_pdb(mut_pdb_path)
        elif re.match(r'WT(.*)bj3.pdb', pdb_file):
            wt_pdb_path = os.path.join(pdb_path, pdb_file)
            print(wt_pdb_path)
            data_wt = parse_pdb(wt_pdb_path)

    print(data_wt['aa'].shape[0], data_mut['aa'].shape[0], ddG)
    return data_wt, data_mut, ddG, mutations



def load_input_csv(input_path: str ="/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Helixon/Helixon_all.csv"):
    #fields = ['#Pdb', 'Mutation(s)_cleaned', 'Affinity_wt_parsed', 'Affinity_mut_parsed']
    fields = ['pdb_code','antibody_heavy_chain_id','antibody_light_chain_id','antigen_chain_id', 'mutations', 'delta_delta_g']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields, delimiter=',')

    res = list()
    mut_info = list()

    for idx, row in df.iterrows():
        if pd.isna(row['delta_delta_g']):
            continue
        #data_wt, data_mut, ddG, mutations = process_row(row)

        try:
            data_wt, data_mut, ddG, mutations = process_rosetta_row(row)
            if not data_wt['aa'].shape[0] == data_mut['aa'].shape[0]: # mutation sequence is not with the same length than before
                continue
            if 'ppi_code' in data_mut.keys() and not data_wt['aa'].shape[0] == data_mut['ppi_code'].shape[0]:
                print("{} ppi code generate error!".format(row['pdb_code']))
                continue
            if torch.all((data_wt['aa'] == data_mut['aa'])): # skip no mutation sequence
                print(torch.all((data_wt['aa'] == data_mut['aa'])))
                continue
            #print(data_wt)
            res.append((data_wt, data_mut, ddG))
            mut_info.append(mutations)
        except:
            continue
    return res, mut_info


def load_Independent_input(input_path: str ="/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/Independent/mutations.txt"):
    fields = ['pdb_code','antibody_heavy_chain_id','antibody_light_chain_id','antigen_chain_id', 'mutations', 'sample_id']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields, delimiter='\t')

    res = list()
    mut_info = list()

    for idx, row in df.iterrows():
        #data_wt, data_mut, ddG, mutations = process_row(row)
        data_wt, data_mut, ddG, _ = process_row(row)
        if not data_wt['aa'].shape[0] == data_mut['aa'].shape[0]: # mutation sequence is not with the same length than before
            raise  RuntimeError("wildtype and mutant do not have same length!")
            continue
        if 'ppi_code' in data_mut.keys() and not data_wt['aa'].shape[0] == data_mut['ppi_code'].shape[0]:
            print("{} ppi code generate error!".format(row['pdb_code']))
            continue
        if torch.all((data_wt['aa'] == data_mut['aa'])): # skip no mutation sequence
            print(torch.all((data_wt['aa'] == data_mut['aa'])))
            continue
        #print(data_wt)
        res.append((data_wt, data_mut, ddG))
        mut_info.append(row['sample_id'])

    return res, mut_info




def save_input_pk(res, save_path: str="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk"):
    with open(save_path, 'wb') as out_file:
        pickle.dump(res, out_file)
        out_file.close()


if __name__ == '__main__':
    """
    res, mut_info = load_input_csv("/ceph/bbkjfeng/buddy2/TaijiPool/bbkjfeng/datasets/SKEMPI2/SKEMPI2_all.csv")
    print(len(res))
    save_input_pk(res, save_path="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi_rosetta_bbk.pk")
    """
    res = load_Independent_input()
    print(len(res[0]))
    save_input_pk(res, save_path="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Independent_bbk.pk")

    """
    #with open("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi_bbk.pk", 'rb') as in_file:
    with open("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/Helixon_bbk.pk", 'rb') as in_file:
        res, _ = pickle.load(in_file)
        in_file.close()
        for (data_wt, data_mut, ddG) in res:
            print(data_wt['aa'].shape[0], data_wt['ppi_code'].shape[0], data_wt['phys'].shape[0], len(data_wt['chain_id']))
            assert (data_wt['aa'].shape[0] == data_wt['ppi_code'].shape[0]), "inconsistent chain length"
    """