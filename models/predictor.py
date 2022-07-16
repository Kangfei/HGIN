import torch
import torch.nn as nn
import torch.nn.functional as F

from models.residue import PerResidueEncoder
from models.gat import GATEncoder
from models.egnn import EGNNEncoder, SimpleReadoutAttention, EGNNEncoderV2
from models.attention import GAEncoder
from models.gvp import GeometricGVPMPNN, GeoGVPMPNN
from models.common import get_pos_CB, construct_3d_basis
from utils.protein import ATOM_N, ATOM_CA, ATOM_C


class ComplexEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.relpos_embedding = nn.Embedding(cfg.max_relpos*2+2, cfg.pair_feat_dim)
        self.residue_encoder = PerResidueEncoder(cfg.node_feat_dim)
        self.residue_encoder = EGNNEncoder(feat_dim=cfg.node_feat_dim, depth= cfg.num_egnn_layers, use_chain_feat=cfg.use_chain_feat) \
            if cfg.res_encoder == 'egnn' else PerResidueEncoder(cfg.node_feat_dim)
        #self.residue_encoder = GATEncoder(feat_dim=cfg.node_feat_dim, depth=cfg.num_egnn_layers)
        #self.residue_encoder = SE3Encoder(feat_dim=cfg.node_feat_dim)

        #self.residue_crg_emb = nn.Embedding(3, 8)
        #self.residue_phys_emb = nn.Linear(2, 8)
        """
        if cfg.geomattn is not None:
            self.ga_encoder = GAEncoder(
                node_feat_dim = cfg.node_feat_dim,
                pair_feat_dim = cfg.pair_feat_dim,
                num_layers = cfg.geomattn.num_layers,
                spatial_attn_mode = cfg.geomattn.spatial_attn_mode,
            )
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim),
            )
        """

        self.ga_encoder_gvp = GeometricGVPMPNN(feats_node= cfg.node_feat_dim,
                                               vectors_node=64,
                                               feats_edge=cfg.pair_feat_dim,
                                               vectors_edge=64)
        """
        self.ga_encoder_gvp = GeoGVPMPNN(feats_node=cfg.node_feat_dim,
                                               vectors_node=64,
                                               feats_edge=cfg.pair_feat_dim,
                                               vectors_edge=64)
        """

    #def forward(self, pos14, aa, seq, chain, mask_atom):
    def forward(self, pos14, aa, seq, phys, crg, chain, mask_atom):
        """
        Args:
            pos14:  (N, L, 14, 3).
            aa:     (N, L).
            seq:    (N, L).
            chain:  (N, L).
            mask_atom:  (N, L, 14)
            phys: (N, L, 2)
            crg: (N, L)
        Returns:
            (N, L, node_ch)
        """
        same_chain = (chain[:, None, :] == chain[:, :, None])   # (N, L, L)
        relpos = (seq[:, None, :] - seq[:, :, None]).clamp(min=-self.cfg.max_relpos, max=self.cfg.max_relpos) + self.cfg.max_relpos # (N, L, L)
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.cfg.max_relpos*2+1)) # (N, L, L)
        pair_feat = self.relpos_embedding(relpos)   # (N, L, L, pair_ch)
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])

        # Residue encoder
        # aa: [N, L], pos14: [N, L, 14, 3], mask_atom: [N, L, 14]
        res_feat = self.residue_encoder(aa, pos14, mask_atom, chain) # (N, L, F)
        #res_feat = self.residue_encoder(aa, pos14, mask_atom, phys, crg, chain)  # (N, L, F)
        """
        phys_feat = self.residue_phys_emb(phys) # (N, L, 16)
        crg_feat = self.residue_crg_emb(crg)  # (N, L, 16)
        res_feat = torch.cat([res_feat, phys_feat, crg_feat], dim=-1)
        """

        # Geom encoder
        t = pos14[:, :, ATOM_CA]
        mask_residue = mask_atom[:, :, ATOM_CA]
        #print("mask_res", mask_residue)
        #res_feat = self.ga_encoder(R, t, get_pos_CB(pos14, mask_atom), res_feat, pair_feat, mask_residue)
        res_feat = self.ga_encoder_gvp(R, t, get_pos_CB(pos14, mask_atom), res_feat, pair_feat, mask_residue)
        #res_feat = self.ga_encoder_gvp(pos14, mask_atom, res_feat, pair_feat, mask_residue)

        return res_feat


class DDGReadout(nn.Module):

    def __init__(self, feat_dim, mode):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.project = nn.Linear(feat_dim, 1, bias=False)
        self.project_cla = nn.Linear(feat_dim, 4, bias=False) if mode == 'cla' else None
        self.project_gau = nn.Linear(feat_dim, 2, bias=False) if mode == 'gau' else None
        self.dropout = nn.Dropout(p =0.5)# for multisample dropout
        self.mode = mode


    def forward(self, node_feat_wt, node_feat_mut, mask=None):
        """
        Args:
            node_feat_wt:   (N, L, F).
            node_feat_mut:  (N, L, F).
            mask:   (N, L).
        """
        feat_wm = torch.cat([node_feat_wt, node_feat_mut], dim=-1)
        feat_mw = torch.cat([node_feat_mut, node_feat_wt], dim=-1)
        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)       # (N, L, F)

        # feat_diff = self.mlp(node_feat_wt) - self.mlp(node_feat_mut)
        if self.mode == 'cla':
            per_residue_ddg = self.project_cla(feat_diff)  # (N, L, 4)
        elif self.mode == 'reg':
            per_residue_ddg = self.project(feat_diff).squeeze(-1)   # (N, L)
            #per_residue_ddg = torch.mean(
            #    torch.stack([self.project(self.dropout(feat_diff)).squeeze(-1) for _ in range(5)],dim=0,),
            #    dim=0,
            #)
        else:
            per_residue_ddg = self.project_gau(feat_diff)  # (N, L, 2)
        if mask is not None:
            #print("per_residue_ddg", per_residue_ddg.shape)
            #print("mask", mask.shape)
            if self.mode == 'cla' or self.mode == 'gau':
                mask = mask[:, :, None].expand_as(per_residue_ddg)
            per_residue_ddg = per_residue_ddg * mask
        ddg = per_residue_ddg.sum(dim=1)    # (N,) for 'reg' or (N, 4) for 'cla' or (N, 4) for 'gau'
        return ddg


class DDGAttentionReadout(nn.Module):

    def __init__(self, feat_dim, mode):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.project = nn.Sequential(
            nn.Linear(feat_dim * 4, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, 1, bias=False)
        )
        self.project_cla = nn.Linear(feat_dim, 4, bias=False) if mode == 'cla' else None
        self.project_gau = nn.Linear(feat_dim, 2, bias=False) if mode == 'gau' else None
        self.att_project = SimpleReadoutAttention(n_hidden=feat_dim, v_hidden=feat_dim, n_expert=4)
        self.mode = mode


    def forward(self, node_feat_wt, node_feat_mut, mask=None):
        """
        Args:
            node_feat_wt:   (N, L, F).
            node_feat_mut:  (N, L, F).
            mask:   (N, L).
        """
        feat_wm = torch.cat([node_feat_wt, node_feat_mut], dim=-1)
        feat_mw = torch.cat([node_feat_mut, node_feat_wt], dim=-1)
        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)       # (N, L, F)

        # feat_diff = self.mlp(node_feat_wt) - self.mlp(node_feat_mut)
        if self.mode == 'cla':
            per_residue_ddg = self.project_cla(feat_diff)  # (N, L, 4)
        elif self.mode == 'reg':
            feat_diff = self.att_project(feat_diff)
            per_residue_ddg = self.project(feat_diff).squeeze(-1)   # (N,)
            #per_residue_ddg = torch.mean(
            #    torch.stack([self.project(self.dropout(feat_diff)).squeeze(-1) for _ in range(5)],dim=0,),
            #    dim=0,
            #)
        else:
            per_residue_ddg = self.project_gau(feat_diff)  # (N, L, 2)
        ddg = per_residue_ddg
        return ddg


class DDGPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = ComplexEncoder(cfg)
        self.ddG_readout = DDGReadout(cfg.node_feat_dim, cfg.mode)
        #self.ddG_readout = DDGAttentionReadout(cfg.node_feat_dim, cfg.mode)

    def forward(self, batch):
        complex_wt = batch['wt']
        complex_mut = batch['mut']
        mask_atom_wt  = complex_wt['pos14_mask'].all(dim=-1)    # (N, L, 14)
        mask_atom_mut = complex_mut['pos14_mask'].all(dim=-1)
        #print(complex_wt['pos14'].shape, complex_wt['aa'].shape, complex_wt['seq'].shape, complex_wt['chain_seq'].shape, mask_atom_wt.shape)
        #feat_wt  = self.encoder(complex_wt['pos14'], complex_wt['aa'], complex_wt['seq'], complex_wt['chain_seq'], mask_atom_wt) # (N, L, 128)
        #feat_mut = self.encoder(complex_mut['pos14'], complex_mut['aa'], complex_mut['seq'], complex_mut['chain_seq'], mask_atom_mut) # (N, L, 128)
        feat_wt = self.encoder(complex_wt['pos14'], complex_wt['aa'], complex_wt['seq'], complex_wt['phys'], complex_wt['crg'], complex_wt['chain_seq'],
                               mask_atom_wt)  # (N, L, 128)
        feat_mut = self.encoder(complex_mut['pos14'], complex_mut['aa'], complex_mut['seq'],  complex_mut['phys'], complex_mut['crg'], complex_mut['chain_seq'],
                                mask_atom_mut)  # (N, L, 128)

        mask_res = mask_atom_wt[:, :, ATOM_CA]
        ddG_pred = self.ddG_readout(feat_wt, feat_mut, mask_res)  # One mask is enough

        return ddG_pred


class FitnessReadout(nn.Module):
    def __init__(self, feat_dim, mode):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.project = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, node_feat_wt, mask=None):
        """
        Args:
            node_feat_wt:   (N, L, F).
            mask:   (N, L).
        """
        feat = self.mlp(node_feat_wt) # (N, L, F)
        per_residue_ddg = self.project(feat).squeeze(-1)  # (N, L)
        if mask is not None:
            per_residue_ddg = per_residue_ddg * mask
        ddg = per_residue_ddg.sum(dim=1) # (N, L)
        return ddg


class FitnessPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = ComplexEncoder(cfg)
        self.fitness_readout = FitnessReadout(cfg.node_feat_dim, cfg.mode)

    def forward(self, batch):
        complex_wt = batch['wt']
        mask_atom_wt = complex_wt['pos14_mask'].all(dim=-1)  # (N, L, 14)
        feat_wt = self.encoder(complex_wt['pos14'], complex_wt['aa'], complex_wt['seq'], complex_wt['chain_seq'],
                               mask_atom_wt)  # (N, L, 128)
        mask_res = mask_atom_wt[:, :, ATOM_CA]
        ddG_pred = self.fitness_readout(feat_wt, mask_res)

        return ddG_pred


class E2EGeoPPIDDGPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = GATEncoder(cfg.node_feat_dim, depth=3, readout='mean_max')
        self.ddG_readout = DDGReadout(cfg.node_feat_dim * 2, cfg.mode)

    def forward(self, batch):
        complex_wt = batch['wt']
        complex_mut = batch['mut']
        mask_atom_wt  = complex_wt['pos14_mask'].all(dim=-1)    # (N, L, 14)
        mask_atom_mut = complex_mut['pos14_mask'].all(dim=-1)
        #print(complex_wt['pos14'].shape, complex_wt['aa'].shape, complex_wt['seq'].shape, complex_wt['chain_seq'].shape, mask_atom_wt.shape)
        feat_wt  = self.encoder( complex_wt['aa'], complex_wt['pos14'], mask_atom_wt) # (N, L, 128)
        feat_mut = self.encoder( complex_mut['aa'], complex_mut['pos14'], mask_atom_mut) # (N, L, 128)

        mask_res = mask_atom_wt[:, :, ATOM_CA]
        ddG_pred = self.ddG_readout(feat_wt, feat_mut, mask_res)  # One mask is enough

        return ddG_pred

class E2EGeoPPIFitnessPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = GATEncoder(cfg.node_feat_dim, depth=3, readout='mean_max')
        self.fitness_readout = FitnessReadout(cfg.node_feat_dim * 2, cfg.mode)

    def forward(self, batch):
        complex_wt = batch['wt']

        mask_atom_wt  = complex_wt['pos14_mask'].all(dim=-1)    # (N, L, 14)

        #print(complex_wt['pos14'].shape, complex_wt['aa'].shape, complex_wt['seq'].shape, complex_wt['chain_seq'].shape, mask_atom_wt.shape)
        feat_wt  = self.encoder( complex_wt['aa'], complex_wt['pos14'], mask_atom_wt) # (N, L, 128)

        mask_res = mask_atom_wt[:, :, ATOM_CA]
        ddG_pred = self.fitness_readout(feat_wt, mask_res)  # One mask is enough

        return ddG_pred