import pandas as pd
import os
import math
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.protein import parse_pdb
from multiprocessing import Pool
import pickle
from scripts.e2e import run_evoef2_build_one_mutant




def process_row(row):
    ddG = 0.59 * math.log(row['Affinity_mut_parsed']) - 0.59 * math.log(row['Affinity_wt_parsed'])
    #print(row['#Pdb'], row['Mutation(s)_cleaned'], ddg)
    pdb_id, p1_chain_ids, p2_chain_ids = row['#Pdb'].split('_')[0], row['#Pdb'].split('_')[1], row['#Pdb'].split('_')[2]
    mutations = '_'.join(row['Mutation(s)_cleaned'].split(','))
    mut_pdb_file = "{}_{}.pdb".format(pdb_id, mutations)
    wt_pdb_file = "{}.pdb".format(pdb_id)
    print(mut_pdb_file, wt_pdb_file)
    mut_pdb_path = os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/mut_pdbs", mut_pdb_file)
    wt_pdb_path = os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/PDBs", wt_pdb_file)
    use_chain_ids = p1_chain_ids + p2_chain_ids
    data_mut = parse_pdb(mut_pdb_path, use_chain_ids)
    data_wt = parse_pdb(wt_pdb_path, use_chain_ids)

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
    assert ppi_code.shape[0] == data_wt['aa'].shape[0], "inconsistent chain length"
    data_wt['ppi_code'] = ppi_code
    data_mut['ppi_code'] = ppi_code

    print(data_wt['aa'].shape[0], data_mut['aa'].shape[0], ddG)
    return data_wt, data_mut, ddG

def load_input_csv(input_path: str ="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi_v2.csv"):
    fields = ['#Pdb', 'Mutation(s)_cleaned', 'Affinity_wt_parsed', 'Affinity_mut_parsed']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields, delimiter=';').dropna() # remove rows contain NA

    res = []
    #p = Pool(32)
    for idx, row in df.iterrows():
        if math.isnan(row['Affinity_mut_parsed']) or math.isnan(row['Affinity_wt_parsed']):
            continue
        try:
            #data_wt, data_mut, ddG = p.apply_async(process_row, args=(row, )).get()
            data_wt, data_mut, ddG = process_row(row)
            if not data_wt['aa'].shape[0] == data_mut['aa'].shape[0]: # mutation sequence is not with the same length than before
                continue
            if 'ppi_code' in data_mut.keys() and not data_wt['aa'].shape[0] == data_mut['ppi_code'].shape[0]:
                print("{} ppi code generate error!".format(row['pdb_code']))
                continue
            if torch.all((data_wt['aa'] == data_mut['aa'])): # skip no mutation sequence
                print(torch.all((data_wt['aa'] == data_mut['aa'])))
                continue
            res.append((data_wt, data_mut, ddG))
        except:
            continue
    #p.close()
    #p.join()
    return res


def load_P36_5D2_ground_truth(neu_path: str, pdb_dir: str, wt_pdb_path:str):
    mut_dict = {'Wildtype': '',
                'Alpha': 'NC501Y',
                'Beta': 'KC417N,EC484K,NC501Y',
                'Gamma': 'KC417T,EC484K,NC501Y',
                'Delta': 'TC478K,LC452R',
                'Delta_Plus': 'KC417T,TC478K,LC452R',
                'Kappa': 'L452R,E484Q',
                'Epsilon': 'LC452R',
                'Eta': 'EC484K',
                'N439K': 'NC439K'}
    res = list()
    mut_info = list()
    data_wt = parse_pdb(wt_pdb_path)
    with open(neu_path, "r") as input_file:
        df = pd.read_excel(neu_path)[2:]

        for idx, row in df.iterrows():
            for mut_name, c_muts in mut_dict.items():
                if row['Antibody'] == 'P36-5D2' and mut_name == 'Wildtype':
                    continue
                if row['Mutation'] == '/':
                    all_muts = c_muts.split(',')
                elif mut_name == 'Wildtype':
                    all_muts = row['Mutation'].split(',')
                else:
                    all_muts = row['Mutation'].split(',') + c_muts.split(',')
                ddG = 0.59 * (math.log(row[mut_name]) - math.log(0.082))
                #print(all_muts, ddG)
                mut_pdb = '{}.pdb'.format('_'.join(all_muts))
                mut_info.append('_'.join(all_muts))
                if not os.path.exists(os.path.join(pdb_dir, mut_pdb)):
                    raise IOError("PDB file {} not exist!".format(mut_pdb))
                data_mut = parse_pdb(os.path.join(pdb_dir, mut_pdb))
                res.append((data_wt, data_mut, ddG))
        input_file.close()
    return res, mut_info


def Evision_mutation_generation(input_path: str ="/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/data/dmsTraining_2017-02-20.csv"):
    fields = ['pdb_id', 'pdb_chain_id', 'aa1', 'aa2', 'position', 'scaled_effect1']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields,  delimiter=',').dropna()  # remove rows contain NA
    evoef2_path='/apdcephfs/share_1364275/kfzhao/Bio_data/EvoEF2'
    mutations_dict = dict()
    for idx, row in df.iterrows():
        if row['pdb_id'] not in mutations_dict.keys():
            mutations_dict[row['pdb_id']] = list()
        pdb_id = row['pdb_id'].lower()
        mut_tag = row['aa1'].strip() + row['pdb_chain_id'].strip() + str(row['position']).strip() + row['aa2'].strip()
        mutations_dict[pdb_id].append(mut_tag)
    for pdb_id, mut_tag_list in mutations_dict.items():
        tmp_work_path = os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Mutants", pdb_id)
        wt_pdb_path = "/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/pdbs/{}.pdb".format(pdb_id)
        if not os.path.exists(os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Mutants", pdb_id)):
            os.makedirs(os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Mutants", pdb_id))

        for mut_tag in mut_tag_list:
            try:
                run_evoef2_build_one_mutant(wt_pdb_file=wt_pdb_path, mut_tags= mut_tag, tmp_work_path= tmp_work_path, evoef2_path= evoef2_path)
                mut_file_name = "{}_{}.pdb".format(pdb_id, mut_tag)
                os.system("mv ./{}_Model_0001.pdb  {}".format(pdb_id, mut_file_name))
                os.system("mv {} {}".format(mut_file_name, os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Mutants", pdb_id)))
                print("success build mutant {} for {}.pdb".format(mut_tag, pdb_id))
            except:
                print("fail build mutant {} for {}.pdb".format(mut_tag, pdb_id))
                continue


def load_Evision_data(input_path: str ="/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/data/dmsTraining_2017-02-20.csv"):
    fields = ['pdb_id', 'pdb_chain_id', 'aa1', 'aa2', 'position', 'scaled_effect1']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields,  delimiter=',').dropna()  # remove rows contain NA
    res = list()
    for idx, row in df.iterrows():

        pdb_id = row['pdb_id'].lower()
        chain_id = row['pdb_chain_id'].strip()
        mut_tag = row['aa1'].strip() + chain_id + str(row['position']).strip() + row['aa2'].strip()
        wt_pdb_file = "/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/pdbs/{}.pdb".format(pdb_id)
        mut_file_name = "{}_{}.pdb".format(pdb_id, mut_tag)
        mut_pdb_file = os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Mutants", pdb_id, mut_file_name)
        if not os.path.exists(mut_pdb_file) or not os.path.exists(wt_pdb_file):
            print("file does not exist")
            continue
        data_wt = parse_pdb(wt_pdb_file, use_chain_ids=chain_id)
        data_mut = parse_pdb(mut_pdb_file, use_chain_ids=chain_id)
        if data_wt is None or data_mut is None:
            print("data load error")
            continue
        if not data_wt['aa'].shape[0] == data_mut['aa'].shape[0]:  # mutation sequence is not with the same length than before
            print(data_wt['aa'].shape[0],  data_mut['aa'].shape[0])
            print("inconsistent length")
            continue
        if torch.all((data_wt['aa'] == data_mut['aa'])):  # skip no mutation sequence
            continue
        #res.append((data_wt, data_mut, row['scaled_effect1']))
        res.append((data_mut, row['scaled_effect1'])) # 'Evision_fitness.pk'
        #print(mut_tag, row['scaled_effect1'])
    return res


def load_input_pk(input_path: str = "/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi.pk"):
    with open(input_path, 'rb') as in_file:
        res = pickle.load(in_file)
        in_file.close()
    return res

def save_input_pk(res, save_path: str="/apdcephfs/share_1364275/kfzhao/Bio_data/PNAS_data/skempi.pk"):
    with open(save_path, 'wb') as out_file:
        pickle.dump(res, out_file)
        out_file.close()


if __name__ == '__main__':


    res = load_Evision_data()
    print(len(res))
    save_input_pk(res, save_path="/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Evision_fitness.pk")
    """
    min_size, max_size = 1000, 0
    for data_wt, data_mut, ddG in res:
        #print(data_wt['aa'].shape[0], data_mut['aa'].shape[0])
        min_size = min(min_size, data_wt['aa'].shape[0])
        max_size = max(max_size, data_wt['aa'].shape[0])
        print(torch.all((data_wt['aa'] == data_mut['aa'])))
    print(min_size, max_size)
    """
    """
    res = load_Evision_data()
    print(len(res))
    save_input_pk(res, save_path="/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Evision.pk")
    """

    """
    res = load_input_pk("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Evision_debug.pk")
    print(len(res))
    cnt = 0
    for (data_wt, data_mut, ddG) in res:
        if data_wt is None or data_mut is None:
            continue
        cnt += 1
    print(cnt)
    """

