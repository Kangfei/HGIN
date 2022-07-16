from models.sequence.rcnn import ResRecCNN
import torch
from torch.nn.modules.module import Module
from torch.nn import Linear
from transformers import BertModel, PreTrainedModel, BertPreTrainedModel


class ProtBertPreTrained(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(BertPreTrainedModel, self).__init__(config)
        self.bert = BertModel(config)
        self.rcnn = ResRecCNN(in_channels=1024, hid_channels=128)


    def forward(self, complex):
        #print(complex)
        seq = complex['residue_seq']
        #print(seq['input_ids'].shape, seq['token_type_ids'].shape, seq['attention_mask'].shape)
        output = self.bert(**seq)
        final_hidden_state = output.hidden_states[-1]  # (B, L, D)
        return final_hidden_state


class BertPIPR(Module):
    def __init__(self, hid_channels, bert_checkpoint: str = "/apdcephfs/share_1364275/kfzhao/Bio_data/model/prot_bert", bert_finetune: bool=False):
        super(BertPIPR, self).__init__()
        self.rcnn = ResRecCNN(1024, hid_channels)
        self.dense = Linear(in_features=hid_channels, out_features=1)
        self.prot_bert_pretrained = ProtBertPreTrained.from_pretrained(bert_checkpoint)
        self.bert_finetune = bert_finetune

    def forward(self, batch):
        complex_wt = batch['wt']
        if self.bert_finetune:
            seq = self.prot_bert_pretrained(complex_wt)
        else:
            with torch.no_grad():
                seq = self.prot_bert_pretrained(complex_wt)
        seq = seq.permute(0, 2, 1)
        seq = self.rcnn(seq)
        x = self.dense(seq).squeeze(dim=-1)
        return x


class BertMuPIPR(Module):
    def __init__(self, hid_channels, bert_checkpoint: str = "/apdcephfs/share_1364275/kfzhao/Bio_data/model/prot_bert", bert_finetune: bool=False):
        super(BertMuPIPR, self).__init__()
        self.rcnn = ResRecCNN(1024, hid_channels)
        self.dense = Linear(in_features=hid_channels, out_features=1)
        self.prot_bert_pretrained = ProtBertPreTrained.from_pretrained(bert_checkpoint)
        self.bert_finetune = bert_finetune

    def forward(self, batch):
        complex_wt, complex_mut = batch['wt'], batch['mut']
        if self.bert_finetune:
            seq1 = self.prot_bert_pretrained(complex_wt)
            seq2 = self.prot_bert_pretrained(complex_mut)
        else:
            with torch.no_grad():
                seq1 = self.prot_bert_pretrained(complex_wt)
                seq2 = self.prot_bert_pretrained(complex_mut)
        seq1, seq2 = seq1.permute(0, 2, 1), seq2.permute(0, 2, 1)
        seq1 = self.rcnn(seq1)
        seq2 = self.rcnn(seq2)
        x = seq1 * seq2
        x = self.dense(x).squeeze(dim=-1)
        return x