# the atom encoder
import torch
from torch import nn
from einops import rearrange, repeat


class AtomEncoder(nn.Module):
    def __init__(self, feat_dim, use_chain_feat= False):
        super(AtomEncoder, self).__init__()
        self.residual_type_emb = nn.Embedding(21, int(feat_dim / 2))
        self.atom_type_emb = nn.Embedding(14, int(feat_dim / 2))
        self.chain_type_emb = nn.Embedding(8, int(feat_dim / 4)) if use_chain_feat else None
        self.final_linear = nn.Linear(feat_dim + int(feat_dim / 4), feat_dim) if use_chain_feat else nn.Identity()
        self.use_chain_feat = use_chain_feat

    def forward(self, aa, pos14, atom_mask, chain_seq = None):
        """
        Args:
            aa:           (N, L).
            pos14:        (N, L, 14, 3).
            atom_mask:    (N, L, 14).
            chain_seq:       (N, L)
        return:
            feats:       (N, L*14, D)
            mask:        (N, L*14)
            coors:    (N, L*14, 3)
        """
        N, L = aa.size()
        residual_feat = aa[:, :, None].expand(N, L, 14).contiguous().view(N, -1).long()
        coors = pos14.view(N, -1, 3)
        mask = atom_mask.view(N, -1)
        atom_feat = torch.arange(0, 14).expand(N, L, 14).contiguous().view(N, -1).long().to(residual_feat)

        residual_feat = self.residual_type_emb(residual_feat)
        atom_feat =  self.atom_type_emb(atom_feat)

        if self.use_chain_feat:
            #print("chain feat", chain_seq)
            chain_feat = chain_seq[:, :, None].expand(N, L, 14).contiguous().view(N, -1).long()
            chain_feat = self.chain_type_emb(chain_feat)
            feats = torch.cat([residual_feat, atom_feat, chain_feat], dim=-1)
        else:
            feats = torch.cat([residual_feat, atom_feat], dim=-1)
        feats = self.final_linear(feats)
        return feats, coors, mask


class AtomEncoderWithPhysChem(nn.Module):
    def __init__(self, feat_dim, use_chain_feat= False):
        super(AtomEncoderWithPhysChem, self).__init__()
        self.residual_type_emb = nn.Embedding(21, int(feat_dim / 2))
        self.atom_type_emb = nn.Embedding(14, int(feat_dim / 2))
        self.chain_type_emb = nn.Embedding(8, int(feat_dim / 4)) if use_chain_feat else None

        self.residue_crg_emb = nn.Embedding(3, 16) # for discrete charge feature
        self.residue_phys_emb = nn.Linear(2, 16) # for continuous feature
        self.final_linear = nn.Linear(feat_dim + int(feat_dim / 4) + 16 + 16, feat_dim) if use_chain_feat \
            else nn.Linear(feat_dim + 16, feat_dim)
        self.use_chain_feat = use_chain_feat


    def forward(self, aa, pos14, atom_mask, phys, crg, chain_seq = None):
        """
        Args:
            aa:           (N, L).
            pos14:        (N, L, 14, 3).
            atom_mask:    (N, L, 14).
            phys:       (N, L, 2)
            crg:        (N, L)
            chain_seq:    (N, L)
        return:
            feats:       (N, L*14, D)
            mask:        (N, L*14)
            coors:    (N, L*14, 3)
        """
        N, L = aa.size()
        residual_feat = aa[:, :, None].expand(N, L, 14).contiguous().view(N, -1).long()
        coors = pos14.view(N, -1, 3)
        mask = atom_mask.view(N, -1)
        atom_feat = torch.arange(0, 14).expand(N, L, 14).contiguous().view(N, -1).long().to(residual_feat)
        #print(residual_feat.shape, atom_feat.shape)

        residual_feat = self.residual_type_emb(residual_feat)
        atom_feat =  self.atom_type_emb(atom_feat)

        phys_feat = phys[:, :, None, :].expand(N, L, 14, 2).contiguous().view(N, -1, 2)
        crg_feat = crg[:, :, None].expand(N, L, 14).contiguous().view(N, -1).long()
        phys_feat = self.residue_phys_emb(phys_feat)
        crg_feat = self.residue_crg_emb(crg_feat)

        if self.use_chain_feat:
            #print("chain feat", chain_seq)
            chain_feat = chain_seq[:, :, None].expand(N, L, 14).contiguous().view(N, -1).long()
            chain_feat = self.chain_type_emb(chain_feat)
            feats = torch.cat([residual_feat, atom_feat, chain_feat, crg_feat], dim=-1)
        else:
            feats = torch.cat([residual_feat, atom_feat, crg_feat], dim=-1)
        feats = self.final_linear(feats)
        return feats, coors, mask


class AtomEncoderWithCoordDistCutoff(nn.Module):
    def __init__(self, feat_dim, used_atom_num=14, max_dist=0.):
        super(AtomEncoderWithCoordDistCutoff, self).__init__()
        self.feat = feat_dim
        res_embed_dim = int((feat_dim - 3) / 2)
        atom_embed_dim = (feat_dim - 3) - res_embed_dim
        self.max_dist = max_dist
        self.residual_type_emb = nn.Embedding(21, res_embed_dim)
        self.atom_type_emb = nn.Embedding(used_atom_num, atom_embed_dim)

    def forward(self, aa, pos14, atom_mask):
        """
            Args:
                aa:           (N, L).
                pos14:        (N, L, 14, 3).
                atom_mask:    (N, L, 14).
            return:
                feats:        (N, L*14, D)
                mask:         (N, L*14)
                edge_mask:    (N, L*14, L*14)
        """
        N, L = aa.size()
        residual_feat = aa[:, :, None].expand(N, L, 14).contiguous().view(N, -1).long()

        mask = atom_mask.view(N, -1)
        atom_feat = torch.arange(0, 14).expand(N, L, 14).contiguous().view(N, -1).long().to(residual_feat)
        # print(residual_feat.shape, atom_feat.shape)

        residual_feat = self.residual_type_emb(residual_feat)
        atom_feat = self.atom_type_emb(atom_feat)

        coors = pos14.view(N, -1, 3)
        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors,
                                                                      'b j d -> b () j d')  # [batch_size, num_node, num_node, 3]
        rel_dist = (rel_coors ** 2).sum(dim=-1,
                                        keepdim=False)  # [batch_size, num_node, num_node] compute the relative distance
        feats = torch.cat([residual_feat, atom_feat, coors], dim=-1)

        edge_mask = (rel_dist < self.max_dist * self.max_dist) if self.max_dist > 0 else None

        return feats, mask, edge_mask

def atom_interaction_mask(pos14, ppi_code, cutoff_dist = 3):
    """
       Args:
           pos14:       (L, 14, 3).
           ppi_code:    (L,).
       return:
           edgemask:    (L*14, L*14)
    """
    L = pos14.size(0)
    coors = pos14.view(-1, 3)
    rel_coors = rearrange(coors, 'i d -> i () d') - rearrange(coors, 'j d -> () j d')  # [num_node, num_node, 3]
    rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=False)  # [num_node, num_node]
    dist_mask = (rel_dist < cutoff_dist * cutoff_dist)  # bool: [num_node, num_node]

    ppi_code = ppi_code[:, None].expand(L, 14).contiguous().view(L * 14)  # [num_node, num_node]
    ppi_mask = torch.logical_xor(rearrange(ppi_code, 'i -> i ()'),
                                 rearrange(ppi_code, 'j -> () j'))  # [num_node, num_node]
    edge_mask = dist_mask & ppi_mask  # 1 to count, 0 do not count
    return edge_mask

def atom_interaction_mask_batch(pos14, ppi_code, cutoff_dist = 3):
    """
       Args:
           pos14:       (N, L, 14, 3).
           ppi_code:    (N, L,).
       return:
           edgemask:    (N, L*14, L*14)
    """
    N, L = pos14.size(0), pos14.size(1)
    coors = pos14.view(N, -1, 3)
    rel_coors = rearrange(coors, 'n i d -> n i () d') - rearrange(coors, 'n j d -> n () j d')  # [N, num_node, num_node, 3]
    rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=False)  # [N, num_node, num_node]
    dist_mask = (rel_dist < cutoff_dist * cutoff_dist)  # bool: [N, num_node, num_node]

    ppi_code = ppi_code[:, :, None].expand(N, L, 14).contiguous().view(N, L * 14)  # [N, num_node, num_node]
    ppi_mask = torch.logical_xor(rearrange(ppi_code, 'n i -> n i ()'),
                                 rearrange(ppi_code, 'n j -> n () j'))  # [N, num_node, num_node]
    edge_mask = dist_mask & ppi_mask  # 1 to count, 0 do not count
    return edge_mask
