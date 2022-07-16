import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Bio.Seq import Seq
from Bio import SeqIO
from utils.protein import parse_pdb
from utils.load import load_input_pk, save_input_pk
import pandas as pd
from scripts.e2e import run_evoef2_build_one_mutant
wt_pdb_file = "/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/pdbs/6r5k.pdb"
pdb_id = '6r5k'
chain_id = 'D'

"""
data = parse_pdb(wt_pdb_file, use_chain_ids=chain_id)
print(data['aa_seq'])
chain_D = Seq(data['aa_seq'])

rrm_fasta = "/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/data/RRM.fasta"
fasta_sequences = SeqIO.parse(open(rrm_fasta),'fasta')

for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    start_pos = chain_D.find(sub=sequence)
    print(start_pos)
    print(sequence)
    print(data['aa_seq'][88 : 88 + len(sequence)]) # find the position at 88
"""

def RRM_mutation_generation(input_path: str = "/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/data/RRM_double.tsv"):
    fields = ['mutation', 'score']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields,  delimiter='\t').dropna()  # remove rows contain NA
    evoef2_path = '/apdcephfs/share_1364275/kfzhao/Bio_data/EvoEF2'
    tmp_work_path = os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/Mutants", "double")
    if not os.path.exists(tmp_work_path):
        os.makedirs(tmp_work_path)

    for idx, row in df.iterrows():
        mut_tags = [ mut_tag[0] + chain_id + str(int(mut_tag[1:-1]) + 87) + mut_tag[-1] for mut_tag in row['mutation'].strip().split(';')]
        try:
            run_evoef2_build_one_mutant(wt_pdb_file=wt_pdb_file, mut_tags=','.join(mut_tags), tmp_work_path=tmp_work_path,
                                        evoef2_path=evoef2_path)
            mut_file_name = "{}_{}.pdb".format(pdb_id, '_'.join(mut_tags))
            os.system("mv ./{}_Model_0001.pdb  {}".format(pdb_id, mut_file_name))
            os.system("mv {} {}".format(mut_file_name, tmp_work_path))
            print("success build mutant {} for {}.pdb".format('_'.join(mut_tags), pdb_id))
        except:
            print("fail build mutant {} for {}.pdb".format('_'.join(mut_tags), pdb_id))
            continue

def load_RRM_data(input_path: str = "/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/data/RRM_double.tsv"):
    fields = ['mutation', 'score']
    df = pd.read_csv(input_path, skipinitialspace=True, usecols=fields,
                     delimiter='\t').dropna()  # remove rows contain NA
    res = []
    for _, row in df.iterrows():
        mut_tags = [ mut_tag[0] + chain_id + str(int(mut_tag[1:-1]) + 87) + mut_tag[-1] for mut_tag in row['mutation'].strip().split(';')]
        mut_file_name = "{}_{}.pdb".format(pdb_id, '_'.join(mut_tags))
        mut_pdb_path = os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/Mutants", "double", mut_file_name)
        data_mut = parse_pdb(mut_pdb_path, use_chain_ids=chain_id)
        for k, val in data_mut.items():
            if k == 'name':
                continue
            data_mut[k] = data_mut[k][88 : 88 + 75]
        res.append((data_mut, row['score']))
    return res



if __name__ == '__main__':
    #RRM_mutation_generation()
    res = load_RRM_data()
    save_input_pk(res, save_path=os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/", "RRM_double.pk"))
    #res = load_input_pk(input_path=os.path.join("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/", "RRM_double.pk"))
    #data_mut, score = res[0]
    #print(len(data_mut['aa']))
    #print(score)




