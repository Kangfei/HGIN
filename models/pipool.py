import torch
from torch import nn, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from models.atom import atom_interaction_mask_batch


class PiPoolingLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(PiPoolingLayer, self).__init__()
        self.fc_1 = nn.Linear(2 * feat_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 1)

    def _get_edge_feats(self, feats):
        feats_i = rearrange(feats, 'b i d -> b i () d')  # [batch_size, num_node, 1, dim]
        feats_j = rearrange(feats, 'b j d -> b () j d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)
        edge_feats = torch.cat((feats_i, feats_j), dim=-1) # (N, L*14, L*14, D)
        #print("edge feats 00:", edge_feats.isnan().all())
        edge_feats = self.fc_1(edge_feats) # (N, L*14, L*14, H)
        #print("fc 1 weights", self.fc_1.weight.isnan().all(), self.fc_1.weight)
        return edge_feats

    def forward(self, feats_wt, feats_mut, pos14_wt, pos14_mut, ppi_code_wt, ppi_code_mut):
        """
        :param feats_wt: (N, L*14, D)
        :param feats_mut: (N, L*14, D)
        :param pos14_wt: (N, L, 14, 3)
        :param pos14_mut: (N, L, 14, 3)
        :param ppi_code_wt: (N, L)
        :param ppi_code_mut: (N, L)
        :param delta_atom_cnt: (N, 14 * 14)
        :return:
        """
        edge_mask_wt = atom_interaction_mask_batch(pos14_wt, ppi_code_wt) # (N, L*14, L*14)
        edge_mask_mut = atom_interaction_mask_batch(pos14_mut, ppi_code_mut) # (N, L*14, L*14)
        edge_feats_wt = self._get_edge_feats(feats_wt)
        edge_feats_mut = self._get_edge_feats(feats_mut)
        #print("edge_feats 0:", edge_feats_wt.isnan().all(), edge_feats_wt)
        edge_feats_wt = edge_feats_wt.masked_fill_(~edge_mask_wt[:, :, :, None].expand_as(edge_feats_wt), 0.)
        edge_feats_mut = edge_feats_mut.masked_fill_(~edge_mask_mut[:, :, :, None].expand_as(edge_feats_mut), 0.)

        feats_diff = edge_feats_mut - edge_feats_wt # (N, L*14, L*14, H)
        feats_cnt = rearrange(feats_diff, "n (l1 c1) (l2 c2) d -> n l1 c1 l2 c2 d", c1=14, c2=14)
        feats_cnt = torch.einsum("n i j k l d->n j l d",feats_cnt)  # (N, 14, 14, H)
        feats_cnt = torch.flatten(self.fc_2(feats_cnt), start_dim=1, end_dim=-1) # (N, 14 * 14)
        return feats_cnt


class PiPoolingSoftmaxLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(PiPoolingSoftmaxLayer, self).__init__()
        self.fc_1 = nn.Linear(2 * feat_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def _get_edge_feats(self, feats):
        feats_i = rearrange(feats, 'b i d -> b i () d')  # [batch_size, num_node, 1, dim]
        feats_j = rearrange(feats, 'b j d -> b () j d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)
        edge_feats = torch.cat((feats_i, feats_j), dim=-1) # (N, L*14, L*14, D)
        #print("edge feats 00:", edge_feats.isnan().all())
        edge_feats = self.fc_1(edge_feats) # (N, L*14, L*14, H)
        #print("fc 1 weights", self.fc_1.weight.isnan().all(), self.fc_1.weight)
        return edge_feats

    def _get_cnt_feats(self, edge_feats, edge_mask):
        edge_feats = edge_feats.masked_fill_(~edge_mask[:, :, :, None].expand_as(edge_feats), 0.)
        cnt_feats = rearrange(edge_feats, "n (l1 c1) (l2 c2) d -> n l1 c1 l2 c2 d", c1=14, c2=14)  # (N, L, 14, L, 14, H)
        cnt_feats = torch.einsum("n i j k l d->n j l d", cnt_feats)  # (N, 14, 14, H)
        # print("feats_cnt", feats_cnt.sum(), feats_cnt.isnan().all())
        cnt_feats = self.fc_2(cnt_feats).squeeze(dim=-1)  # (N, 14, 14)
        cnt_mask = rearrange(edge_mask, "n (l1 c1) (l2 c2) -> n l1 c1 l2 c2", c1=14, c2=14)  # (N, L, 14, L, 14)
        cnt_mask = torch.einsum("n i j k l ->n j l", cnt_mask).bool()  # (N, 14, 14)
        #print("cnt feat, cnt mask", cnt_feats, cnt_mask)

        cnt_feats.masked_fill_(~cnt_mask, float('-1e5')) # (N, 14, 14) can not use '-inf' to mask
        cnt_feats = self.softmax(cnt_feats)
        #print("cnt feat 2", cnt_feats)
        cnt_feats = torch.flatten(cnt_feats, start_dim=1, end_dim=-1) # (N, 14 * 14)
        return cnt_feats

    def forward(self, feats_wt, feats_mut, pos14_wt, pos14_mut, ppi_code_wt, ppi_code_mut):
        """
        :param feats_wt: (N, L*14, D)
        :param feats_mut: (N, L*14, D)
        :param pos14_wt: (N, L, 14, 3)
        :param pos14_mut: (N, L, 14, 3)
        :param ppi_code_wt: (N, L)
        :param ppi_code_mut: (N, L)
        :param delta_atom_cnt: (N, 14 * 14)
        :return:
        """
        edge_mask_wt = atom_interaction_mask_batch(pos14_wt, ppi_code_wt) # (N, L*14, L*14)
        edge_mask_mut = atom_interaction_mask_batch(pos14_mut, ppi_code_mut) # (N, L*14, L*14)
        edge_feats_wt = self._get_edge_feats(feats_wt)
        edge_feats_mut = self._get_edge_feats(feats_mut)
        cnt_feats_wt = self._get_cnt_feats(edge_feats_wt, edge_mask_wt)
        cnt_feats_mut = self._get_cnt_feats(edge_feats_mut, edge_mask_mut)
        return cnt_feats_mut - cnt_feats_wt
