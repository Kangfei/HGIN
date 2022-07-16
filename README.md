# DDG Predictor with EGNN

## Docker

For V100:
```
mirrors.tencent.com/docker_coffeezhao/eghn_ubuntu16.04-cuda10.1-torch1.8.0-dgl0.7.0-pyg1.7.2_biopython_neural_tangents:latest
```
For A100:
```
mirrors.tencent.com/docker_coffeezhao/eghn_ubuntu16.04-cuda11.1-torch1.9.0-dgl0.7.0-pyg1.7.2_biopython_neural_tangents:latest
```
When using the distributed training in parallel/trainer.py, the latest accelerate should be reinstalled as (to be fixed ;) ):

```
pip uninstall accelerate
git clone https://github.com/huggingface/accelerate
cd accelerate
python setup.py install
```


## Usage

### 1.Predict
```bash
python ./scripts/predict.py \
       --model  <model_ckpt_path> \
       --wt_pdb <path-to-wild-type-pdb> \
       --mut_pdb <path-to-mutant-pdb>
```


### 2.Test helixon dataset
```bash
python ./script/test.py \ 
        --model <model_ckpt_path> \
        --neu_path <neutralization_ground-truth_path> \
        --mut_pdb_dir <mut_pdbs_directory> \
        --wt_pdb <wt_pdb_path>
```
### 3.End2end one mutation test
```bash
python ./script/e2e.py \ 
        --model <model_ckpt_path> \
        --evoef2_path <evoef2_installation_path> \
        --wt_pdb <wt_pdb_path> \
        --mut_tags <muation string> \
        --clean_work_path <empty_working_directory>
```

### 4.Training
Single GPU:

```bash
python ./script/train.py \ 
        --model ./data/model.pt \
        --save_ckpt_dir <model_ckpt_dir> \
        --input_data <serialized_training_data> 
```
Distributed:
```bash
python -m torch.distributed.launch --nproc_per_node <num_gpu_to_use>  --use_env --master_port 20654  ./script/train.py \ 
        --model ./data/model.pt \
        --save_ckpt_dir <model_ckpt_dir> \
        --input_data <serialized_training_data> 
```


#### Input format
The input of the personalized '--input_data' is a binary file 
serialized  by pickle, which is a python list of triples
```
(data_wt, data_mut, ddG)
```
where ddG is a float value,  data_wt/data_mut is the return of utils.protein.parse_pdb, which is a dictionary of 

```
 {
        'name': structure.get_id(),

        # Chain info
        'chain_id': ''.join(chain_id), # sequence format of the chain id
        'chain_seq': torch.LongTensor(chain_seq), # chain_id of a residue in, (L, )

        # Sequence
        'aa': torch.LongTensor(aa), # residue type id (L,)
        'aa_seq': ''.join(aa_seq), #  sequence of residue type
        'resseq': torch.LongTensor(resseq), 
        'icode': ''.join(icode), 
        'seq': torch.LongTensor(seq), 
        
        # Atom positions
        'pos14': torch.stack(pos14), # all atom coordinates of a residue (L, 14, 3)
        'pos14_mask': torch.stack(pos14_mask),  # mask flag for empty atom, (L, 14)

        # Physicochemical Property
        'phys': torch.stack(phys), # numerical value property, (L, 2)
        'crg': torch.LongTensor(crg), # residue sidechain charge, (L,)
        ### L is the total number of residues in the protein
   }
```

#### Key Parameters

| name | type   | description | 
| ----- | --------- | ----------- |
| res_encoder | String |  'mlp' for original DDGPredicor, 'egnn' for EGNN encoder       |
| mode  |   String  | 'reg' for MSE loss, 'cla' for Cross Entropy Loss, 'gau' for Gaussian loss  |
| k | Int |  number of neighbors nearby the mutation to be used  |
| num_egnn_layers | Int |  number of EGNN layers, when res_encoder is 'egnn' |
| ckpt_freq | Int |  number of epochs for model checkpoint  |

All the parameters with their default value are in script/train.py

## Citation

Coming soon...

## Contact

Please contact luost[at]helixon.com for any questions related to the source code.
