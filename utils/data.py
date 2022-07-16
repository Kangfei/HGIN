import math
import torch
from torch.utils.data._utils.collate import default_collate

from .protein import ATOM_CA, parse_pdb
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from models.atom import atom_interaction_mask
import numpy as np

class PaddingCollate(object):

    def __init__(self, length_ref_key='mutation_mask', pad_values={'aa': 20, 'pos14': float('999'), 'icode': ' ', 'chain_id': '-'}, donot_pad={'foldx'}, eight=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.donot_pad = donot_pad
        self.eight = eight

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n, "actual size: {}, pad size: {}".format(x.size(0), n)
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        elif isinstance(x, str):
            if value == 0:  # Won't pad strings if not specified
                return x
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, dict):
            padded = {}
            for k, v in x.items():
                if k in self.donot_pad:
                    padded[k] = v
                else:
                    padded[k] = self._pad_last(v, n + 2, value=self._get_pad_value(k)) \
                        if k in ['input_ids', 'token_type_ids', 'attention_mask'] \
                        else self._pad_last(v, n, value=self._get_pad_value(k))
            return padded
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items() if k in ('wt', 'mut', 'ddG', 'mutation_mask', 'index', 'mutation')
            }
            if 'delta_atom_cnt' in data.keys():
                data_padded['delta_atom_cnt'] = data['delta_atom_cnt']
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def _mask_list(l, mask):
    return [l[i] for i in range(len(l)) if mask[i]]


def _mask_string(s, mask):
    return ''.join([s[i] for i in range(len(s)) if mask[i]])


def _mask_dict_recursively(d, mask):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            out[k] = _mask_list(v, mask)
        elif isinstance(v, str) and len(v) == mask.size(0):
            out[k] = _mask_string(v, mask)
        elif isinstance(v, dict):
            out[k] = _mask_dict_recursively(v, mask)
        else:
            out[k] = v
    return out


class KnnResidue(object):

    def __init__(self, num_neighbors=48):
        super().__init__()
        self.num_neighbors = num_neighbors

    def __call__(self, data):
        pos_CA = data['wt']['pos14'][:, ATOM_CA] # coordinates of C_alpha of all residues of wild-type protein  dim: (L, 3)
        pos_CA_mut = pos_CA[data['mutation_mask']] #  # coordinates of C_alpha of mutated residue,  dim: (num_mutation, 3)
        #print("pos_CA:", pos_CA.shape)
        #print("pos_CA_mut:", pos_CA_mut.shape)
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1) # [L, 1], distance from all residue to the mutated residue
        #print("dist:", dist.shape)

        try:
            mask = torch.zeros([dist.size(0)], dtype=torch.bool)
            mask[ dist.min(dim=1)[0].argsort()[:self.num_neighbors] ] = True  # mask: [L]

        except IndexError as e:
            print(data)
            raise e

        return _mask_dict_recursively(data, mask)


class SequenceResidue(object):
    def __init__(self, seq_len=48):
        super().__init__()
        self.seq_len = seq_len

    def __call__(self, data):
        indices = data['mutation_mask'].nonzero().squeeze(-1)
        l, r = indices[0], indices[-1]

        if r - l + 1 >= self.seq_len:
            start, end = l, r
        else:
            start = max(0, l - int((self.seq_len - (r - l + 1)) / 2))
            end = min(r + (self.seq_len - (start - l)), data['mutation_mask'].size(0))
            if start == l:
                end -= 1
            else:
                start += 1
        try:
            mask = torch.zeros([data['mutation_mask'].size(0)], dtype=torch.bool)
            mask[start:end] = True
        except IndexError as e:
            print(data)
        return _mask_dict_recursively(data, mask)


class PPIResidue(object):

    def __init__(self, num_neighbors=48, num_distances=48):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.num_distances = num_distances

    def __call__(self, data):
        pos_CA = data['wt']['pos14'][:, ATOM_CA]  # coordinates of C_alpha of all residues of wild-type protein  dim: (L, 3)
        ppi_code = data['wt']['ppi_code']
        #print("ppi_code", ppi_code.shape, 'aa', data['wt']['aa'].shape)

        rel_coors = rearrange(pos_CA, 'i d -> i () d') - rearrange(pos_CA, 'j d -> () j d')  # [L, L, 3]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=False)  # [L, L] residue relative distance

        rel_code = torch.logical_xor(rearrange(ppi_code, 'i -> i ()'), rearrange(ppi_code, 'j -> () j')) # [ L, L]
        #print(rel_code)
        rel_code_mask = torch.where(rel_code, 0, 10000000)
        #print(rel_code_mask.shape, rel_dist.shape)
        rel_dist += rel_code_mask

        values, indices = torch.topk(rel_dist.flatten(), k = self.num_distances, largest=False) # find the k-th smallest distances
        indices = (np.array(np.unravel_index(indices.numpy(), rel_dist.shape)).T)
        merged_indices = list(set(list(indices[:, 0]) + list(indices[:, 1]))) # node indices involved in the k-th smallest distances

        #print(ppi_code[merged_indices])

        pos_CA_mut = pos_CA[merged_indices]  # (Q, 3) coordinates of the residue in the PPI interface
        # print("pos_CA:", pos_CA.shape)
        # print("pos_CA_mut:", pos_CA_mut.shape)
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1)  # [L, 1], distance from all residue to the mutated residue
        # print("dist:", dist.shape)

        try:
            mask = torch.zeros([dist.size(0)], dtype=torch.bool)
            mask[dist.min(dim=1)[0].argsort()[:self.num_neighbors]] = True  # mask: [L]
            #print("mut hits", torch.dot(mask.long(), data['mutation_mask'].long()),
            #      torch.sum(data['mutation_mask'].long()))
            #print(ppi_code[mask])

        except IndexError as e:
            print(data)
            raise e

        return _mask_dict_recursively(data, mask)


def load_wt_mut_pdb_pair(wt_path, mut_path):

    data_wt = parse_pdb(wt_path)
    data_mut = parse_pdb(mut_path)

    transform = KnnResidue()
    collate_fn = PaddingCollate()
    mutation_mask = (data_wt['aa'] != data_mut['aa']) # flag indicate the difference of residual type # [L]

    batch = collate_fn([transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask})])
    print(batch['wt']['aa_seq'])
    print(batch['mut']['aa_seq'])
    print(batch['mutation_mask'])
    return batch


# Dataset added by kfzhao
class DDGDataset(Dataset):
    '''
    Dataset for a pair protein  (data_wt, data_mut) to predict the score 'ddG'
    '''
    def __init__(self, data, num_neighbors: int = 48, mode:str = 'reg', tokenizer = None):
        self.transform = KnnResidue(num_neighbors=num_neighbors) if not tokenizer else SequenceResidue(seq_len=num_neighbors)
        self.data = data
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_wt, data_mut, ddG = self.data[item]
        mutation_mask = (data_wt['aa'] != data_mut['aa']) # flag indicate the difference of residual type # [L]
        if self.mode == 'cla': # generate the class label for classification task {0, 1, 2, 3}
            if ddG >= 1.0:
                label = 0
            elif 0 <= ddG < 1.0:
                label = 1
            elif -1 <= ddG < 0:
                label = 2
            else:
                label = 3
        else:
            label = ddG
        data = self.transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask, 'ddG': label})
        if self.tokenizer:
            data['wt']['residue_seq'] = self.tokenizer(data['wt']['aa_seq'])  # Transform the sequence to tenor
            data['mut']['residue_seq'] = self.tokenizer(data['mut']['aa_seq'])
            #print(data_wt['residue_seq']['input_ids'].shape, len(data_wt['aa_seq']))
            #print(data_mut['residue_seq']['input_ids'].shape)
        return data


# Dataset added by kfzhao
class DDGDatasetWithPiPool(Dataset):
    '''
    Dataset for a pair protein  (data_wt, data_mut) to predict the score 'ddG'
    '''
    def __init__(self, data, num_neighbors: int = 48, mode:str = 'reg', tokenizer = None):
        self.transform = KnnResidue(num_neighbors=num_neighbors) if not tokenizer else SequenceResidue(seq_len=num_neighbors)
        self.data = data
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def _atom_interaction_cnt(self, data_wt, cutoff_dist = 3, normalize = False):
        pos14, ppi_code, pos14_mask = data_wt['pos14'], data_wt['ppi_code'], data_wt['pos14_mask']
        edge_mask = atom_interaction_mask(pos14, ppi_code, cutoff_dist)
        L = pos14.size(0)
        mask_atom_wt = pos14_mask.all(dim=-1)  # (L, 14)
        atom = mask_atom_wt.view(L * 14, 1).contiguous().long()
        atom_cnt = torch.matmul(atom, atom.t()).masked_fill_(~edge_mask, 0)
        atom_cnt = rearrange(atom_cnt, "(l1 c1) (l2 c2) -> l1 c1 l2 c2", c1=14, c2=14)
        atom_cnt = torch.einsum("i j k l->j l", atom_cnt)  # (14, 14)
        if normalize:
            atom_cnt = atom_cnt / (torch.sum(atom_cnt) + 1e-6)
        return atom_cnt

    def __getitem__(self, item):
        data_wt, data_mut, ddG = self.data[item]
        mutation_mask = (data_wt['aa'] != data_mut['aa']) # flag indicate the difference of residual type # [L]
        if self.mode == 'cla': # generate the class label for classification task {0, 1, 2, 3}
            if ddG >= 1.0:
                label = 0
            elif 0 <= ddG < 1.0:
                label = 1
            elif -1 <= ddG < 0:
                label = 2
            else:
                label = 3
        else:
            label = ddG
        data = self.transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask, 'ddG': label})
        if self.tokenizer:
            data['wt']['residue_seq'] = self.tokenizer(data['wt']['aa_seq'])  # Transform the sequence to tenor
            data['mut']['residue_seq'] = self.tokenizer(data['mut']['aa_seq'])
            #print(data_wt['residue_seq']['input_ids'].shape, len(data_wt['aa_seq']))
            #print(data_mut['residue_seq']['input_ids'].shape)
        atom_cnt_wt = self._atom_interaction_cnt(data['wt'], normalize= False)
        atom_cnt_mut = self._atom_interaction_cnt(data['mut'], normalize= False)
        data['delta_atom_cnt'] = torch.flatten(atom_cnt_mut - atom_cnt_wt)
        #print(data['delta_atom_cnt'].shape)
        return data


class FitnessDataset(Dataset):
    """
       Dataset for one protein  'data_wt' to predict the score 'ddG'
       """
    def __init__(self, data, mode: str = 'reg', tokenizer = None):
        self.data = data
        self.mode = mode
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_wt, ddG = self.data[item]
        if self.tokenizer:
            data_wt['residue_seq'] = self.tokenizer(data_wt['aa_seq']) # Transform the sequence to tenor
        if self.mode == 'cla': # generate the class label for classification task {0, 1, 2, 3}
            if ddG >= 1.0:
                label = 0
            elif 0 <= ddG < 1.0:
                label = 1
            elif -1 <= ddG < 0:
                label = 2
            else:
                label = 3
        else:
            label = ddG
        mutation_mask = torch.zeros([data_wt['aa'].size(0)], dtype=torch.bool)
        return {'wt': data_wt, 'ddG': label, 'mutation_mask': mutation_mask}

if __name__ == '__main__':
    import pickle
    with open("/apdcephfs/share_1364275/kfzhao/Bio_data/RRM_data/RRM_single.pk", 'rb') as in_file:
        res = pickle.load(in_file)
        in_file.close()
        dataset = FitnessDataset(res)

        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate())
        for step, batch in enumerate(data_loader):
            print(batch)