# refer to the IMPR2 of https://github.com/gordicaleksa/pytorch-GAT/blob/main/models/definitions/GAT.py
import torch
from torch import nn
from einops import rearrange, repeat
from models.atom import *


class GAT(nn.Module):
    def __init__(self,
                 dim,
                 num_heads = 8,
                 dropout = 0.6,
                 bias = True,
                 norm_feats = False,
                 skip_connection = True):
        super(GAT, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.skip_connection = skip_connection

        self.linear_proj = nn.Linear(dim, dim * num_heads, bias = False)
        self.scoring_fn_src = nn.Parameter(torch.Tensor(1, num_heads, dim))
        self.scoring_fn_tag = nn.Parameter(torch.Tensor(1, num_heads, dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.leaklyReLU = nn.LeakyReLU(0.2) # using 0.2 as in the paper
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(p = dropout)
        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity()
        self.__init__parameters()

    def __init__parameters(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_src)
        nn.init.xavier_uniform_(self.scoring_fn_tag)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, feats, mask = None, edge_mask = None):
        """
        :param feats: (N, L, dim)
        :param mask: (N, L)
        :return:
        """
        bs, feats_in = feats.size(0), feats
        if mask is not None and edge_mask is None:
            mask_i = rearrange(mask, 'b i -> b i ()')
            mask_j = rearrange(mask, 'b j -> b () j')
            edge_mask = mask_i * mask_j # (N, L, L)

        #feats = self.dropout(feats) # (N, L, D)
        feats = self.linear_proj(feats).view(bs, -1, self.num_heads, self.dim) # (N, L, H, OD)
        if mask is not None:
            feats = feats.masked_fill_(~mask[:,:, None, None].expand_as(feats), 0.)
        feats = self.dropout(feats)

        score_src = torch.einsum('nlhd,ihd->nlhi', feats, self.scoring_fn_src) # (N, L, H, 1)
        score_tag = torch.einsum('nlhd,ihd->nlhi', feats, self.scoring_fn_tag)


        score_src = score_src.permute(0, 2, 1, 3)  # (N, H, L, 1)
        score_tag = score_tag.permute(0, 2, 3, 1)  # (N, H, 1, L)
        all_scores = self.leaklyReLU(score_src + score_tag) # (N, H, L, L)

        if edge_mask is not None:
            all_scores = all_scores.masked_fill_(~edge_mask[:, None,:, :].expand_as(all_scores), float('-inf'))
            #print("all_scores", all_scores)


        all_att_coef = self.softmax(all_scores) # (N, H, L, L)
        #print("all_att_coef", all_att_coef)

        feats_out = torch.einsum('nhll, nhld->nhld', all_att_coef, feats.permute(0, 2, 1, 3)) # (N, H, L, D)
        feats_out = feats_out.permute(0, 2, 1, 3) # (N, L, H, D)
        feats_out = self.node_norm(feats_out)
        if self.skip_connection:
            feats_out += feats_in.unsqueeze(dim=2) # (N, L, 1, D) + (N, L, H, D) -> (N, L, H, D)
        feats_out = torch.mean(feats_out, dim=2, keepdim=False) # (N, L, D)
        if self.bias is not None:
            feats_out += self.bias

        return feats_out


class GATEncoder(nn.Module):
    def __init__(self, feat_dim, depth, readout = 'sum'):
        super(GATEncoder, self).__init__()
        assert readout in ['sum', 'mean_max'], "Unsupported readout layer"
        self.depth = depth
        self.readout = readout
        self.atom_encoder = AtomEncoderWithCoordDistCutoff(feat_dim, max_dist=3.0)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(GAT(dim=feat_dim, norm_feats=True))

    def forward(self, aa, pos14, atom_mask, chain = None):
        """
            Args:
                aa:           (N, L).
                pos14:        (N, L, 14, 3).
                atom_mask:    (N, L, 14).
        """
        N, L = aa.size()
        feats, mask, edge_mask = self.atom_encoder(aa, pos14, atom_mask)
        #print(feats.shape)
        for layer in self.layers:
            feats = layer(feats, mask, edge_mask)
        mask_out = mask[:, :, None].expand_as(feats)
        # print(mask_out, mask_out.shape)
        feats = torch.where(mask_out, feats, torch.zeros_like(feats))
        if self.readout == 'sum':
            feats = feats.view(N, L, 14, -1).sum(dim=-2) # (N, L, D)
        else:
            feats_mean = torch.mean(feats.view(N, L, 14, -1), dim=-2) # (N, L, D)
            feats_max, _ = torch.max(feats.view(N, L, 14, -1), dim=-2)  # (N, L, D)

            feats = torch.cat([feats_max, feats_mean], dim=-1) # (N, L, 2D)
        #print("feats", torch.isnan(feats).any())

        return feats
