import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data import FitnessDataset, DDGDataset ,PaddingCollate
from torch.utils.data import DataLoader
from models.sequence.rcnn import MuPIPR, PIPR
from models.sequence.bert import BertPIPR, BertMuPIPR
from transformers import BertTokenizer, AlbertTokenizer




class Word2VecTokenizer(object):

    def __init__(self, filename: str= "/apdcephfs/private_coffeezhao/PycharmProjects/seq_ppi/embeddings/string_vec7.txt"):
        self.t2v = {}
        self.dim = None
        for line in open(filename):
            line = line.strip().split('\t')
            t = line[0]
            v = torch.FloatTensor([float(x) for x in line[1].split()])
            if self.dim is None:
                self.dim = len(v)
            else:
                v = v[:self.dim]
            self.t2v[t] = v

    def __call__(self, seq):
        rst = []
        for x in seq:
            v = self.t2v.get(x)
            if v is None:
                continue
            rst.append(v)
        return torch.cat(rst, dim=0).reshape(-1, self.dim)

class BertLMTokenizer(object):
    def __init__(self, bert_type: str= 'prot_bert', filename : str= "/apdcephfs/share_1364275/kfzhao/Bio_data/model/prot_bert"):
        if bert_type == 'prot_albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(filename, do_lower_case=False)

        elif bert_type == 'prot_bert':
            self.tokenizer = BertTokenizer.from_pretrained(filename, do_lower_case=False)
        else:
            raise NotImplementedError("Unsupported model name!")

    def __call__(self, seq):
        seq = " ".join(seq)
        output = {k : torch.LongTensor(v) for k, v in self.tokenizer(seq).items()}

        return output

if __name__ == '__main__':
    import pickle
    tokenizer = Word2VecTokenizer(filename="/apdcephfs/private_coffeezhao/PycharmProjects/seq_ppi/embeddings/string_vec7.txt")
    #tokenizer = BertLMTokenizer()



    with open("/apdcephfs/share_1364275/kfzhao/Bio_data/Evision_data/Evision.pk", 'rb') as in_file:
        res = pickle.load(in_file)
        #print(len(res))
        in_file.close()

        #dataset = FitnessDataset(data=res, tokenizer=tokenizer)
        dataset = DDGDataset(data=res, tokenizer=tokenizer)
        #model = PIPR(in_channels=7, hid_channels=128)
        model = MuPIPR(in_channels=7, hid_channels=128)
        #model = BertPIPR(hid_channels=128)
        #model = BertMuPIPR(hid_channels=128)
        model.train()
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate())
        for step, batch in enumerate(data_loader):
            #print(batch['wt']['residue_seq'])
            output = model(batch)
            print("output:", output)




