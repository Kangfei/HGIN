import torch
from torch import nn, einsum
from models.egnn import exists

from einops import rearrange, repeat
from models.common import global_to_local, dihedrals, orientations, sidechains, rbf, positional_embeddings, normalize_vector, get_pos_CA

class GVP(nn.Module):
    def __init__(
        self,
        *,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        feats_activation = nn.Sigmoid(),
        vectors_activation = nn.Sigmoid(),
        vector_gating = False
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out)

        self.Wh = nn.Parameter(torch.randn(dim_vectors_in, dim_h))
        self.Wu = nn.Parameter(torch.randn(dim_h, dim_vectors_out))

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + dim_feats_in, dim_feats_out),
            feats_activation
        )

        # branching logic to use old GVP, or GVP with vector gating

        self.scalar_to_vector_gates = nn.Linear(dim_feats_out, dim_vectors_out) if vector_gating else None

    def forward(self, data):
        feats, vectors = data
        v, c = vectors.size(-2), vectors.size(-1)
        n = feats.size(-1)

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('... v c, v h -> ... h c', vectors, self.Wh)
        Vu = einsum('... h c, h u -> ... u c', Vh, self.Wu)

        sh = torch.norm(Vh, p = 2, dim = -1)

        s = torch.cat((feats, sh), dim = -1)

        feats_out = self.to_feats_out(s)

        if exists(self.scalar_to_vector_gates):
            gating = self.scalar_to_vector_gates(feats_out)
            gating = gating.unsqueeze(dim = -1)
        else:
            gating = torch.norm(Vu, p = 2, dim = -1, keepdim = True)

        vectors_out = self.vectors_activation(gating) * Vu
        return (feats_out, vectors_out)


class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super().__init__()
        self.vector_dropout = nn.Dropout2d(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)


class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):
        vector_norm = vectors.norm(dim=(-1,-2), keepdim=True)
        normed_feats = self.feat_norm(feats)
        normed_vectors = vectors / (vector_norm + self.eps)
        return normed_feats, normed_vectors


class GVPMPNN(nn.Module):
    """ GVP MPNN layer for Protein"""
    def __init__(self, feats_node, vectors_node,
                 feats_edge, vectors_edge,
                 dropout, residual = False, agg = "mean"):
        super(GVPMPNN, self).__init__()
        self.feats_node, self.vectors_node = feats_node, vectors_node
        self.feats_edge, self.vectors_edge = feats_edge, vectors_edge
        self.dropout = dropout
        self.norm = nn.ModuleList([GVPLayerNorm(self.feats_node),  # + self.feats_edge_out
                                   GVPLayerNorm(self.feats_node)])
        self.dropout = GVPDropout(dropout)
        self.agg = agg
        self.residual = residual
        #  this receives the vec_in message AND the receiver node
        self.W_EV = nn.Sequential(GVP(
            dim_vectors_in=self.vectors_node + self.vectors_edge,
            dim_vectors_out=self.vectors_node + self.feats_edge,
            dim_feats_in=self.feats_node + self.feats_edge,
            dim_feats_out=self.feats_node + self.feats_edge
        ),
            GVP(
                dim_vectors_in=self.vectors_node + self.feats_edge,
                dim_vectors_out=self.vectors_node + self.feats_edge,
                dim_feats_in=self.feats_node + self.feats_edge,
                dim_feats_out=self.feats_node + self.feats_edge
            ),
            GVP(
                dim_vectors_in=self.vectors_node + self.feats_edge,
                dim_vectors_out=self.vectors_node + self.feats_edge,
                dim_feats_in=self.feats_node + self.feats_edge,
                dim_feats_out=self.feats_node + self.feats_edge
            ))

        self.W_dh = nn.Sequential(GVP(
            dim_vectors_in=self.vectors_node,
            dim_vectors_out=2 * self.vectors_node,
            dim_feats_in=self.feats_node,
            dim_feats_out=4 * self.feats_node
        ),
            GVP(
                dim_vectors_in=2 * self.vectors_node,
                dim_vectors_out=self.vectors_node,
                dim_feats_in=4 * self.feats_node,
                dim_feats_out=self.feats_node
            ))

    def forward(self, feats_node, vectors_node, feats_edge, vectors_edge):
        """
        :param feats_node: (N, L, feats_node_in)
        :param vectors_node: (N, L, vectors_node_in)
        :param feats_edge: (N, L, L, feats_edge_in)
        :param vectors_edge: (N, L, L, vectors_edge_in)
        :return:
        """
        # build the messages
        feats_j = repeat(feats_node, 'b j d -> b i j d', i = feats_edge.shape[1])
        vectors_j = repeat(vectors_node, 'b j d c-> b i j d c', i = vectors_node.shape[1])
        # print("feats_j, feats_edge", feats_j.shape, feats_edge.shape)
        feats_message = torch.cat([feats_j, feats_edge], dim=-1) # (N, L, L, D)
        # print("vectors_j, vectors_edge", vectors_j.shape, vectors_edge.shape)
        vectors_message = torch.cat([vectors_j, vectors_edge], dim=-2) # (N, L, L, D, 3)
        feats_message_out, vectors_message_out = self.W_EV( (feats_message, vectors_message) ) # (N, L, L, D), (N, L, L, D, 3)
        # print("feats_message, vectors_message:", feats_message_out.shape, vectors_message_out.shape)
        # perform the aggregation
        if self.agg == "mean":
            feats = feats_message_out.mean(dim=2, keepdim= False)
            vectors = vectors_message_out.mean(dim=2, keepdim = False)
        elif self.agg == "sum":
            feats = feats_message_out.sum(dim = 2, keepdim= False)
            vectors = vectors_message_out.sum(dim = 2, keepdim = False)
        else:
            raise NotImplementedError("Unsupported message aggregation op: {}".format(self.agg))

        #print("feats, vectors", feats.shape, vectors.shape)
        feats, vectors = self.dropout(feats, vectors)
        # get the feats and vectors of the node part
        feats = feats[:, :, :self.feats_node]
        vectors = vectors[:, :, :self.vectors_node, :]
        #print("feats, vectors", feats.shape, vectors.shape)
        feats, vectors = self.norm[0](feats + feats_node, vectors + vectors_node)

        feats_out_, vectors_out_ = self.dropout(* self.W_dh( (feats, vectors) ))
        feats_out, vectors_out = self.norm[1](feats_out_ + feats, vectors_out_ + vectors)

        if self.residual:
            feats_out += feats_node
            vectors_out += vectors_out
        return feats_out, vectors_out


class GeometricGVPMPNN(nn.Module):
    def __init__(self, feats_node, vectors_node,
                 feats_edge, vectors_edge, dropout=0.1, residual = False, num_layers = 3):
        super(GeometricGVPMPNN, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(GVPMPNN(feats_node, vectors_node, feats_edge, vectors_edge, dropout, residual))
        self.vector_node_embed = nn.Linear(1, vectors_node)
        self.vector_edge_embed = nn.Linear(1, vectors_edge)

    def forward(self, R, t, p_CB, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            p_CB: (N, L, 3)
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        """
        feats_node, feats_edge = x, z # (N, L, D), (N, L, L, D)
        #print("x, z", x.shape, z.shape)
        p_CA = t
        vectors_node = global_to_local(R, t, p_CB) # (N, L, 3)
        #print("vectors_node", vectors_node.unsqueeze(-1).shape)
        vectors_node = self.vector_node_embed(vectors_node.unsqueeze(-1)).permute(0, 1, 3, 2) # (N, L, D, 3)
        rel_coors = rearrange(p_CA, 'b i d -> b i () d') - rearrange(p_CA, 'b j d -> b () j d')  # [batch_size, num_node, num_node, 3]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=False)  # [batch_size, num_node, num_node] compute the relative distance

        vectors_edge = einsum('b i j d, b i j -> b i j d', rel_coors, (1.0 / (rel_dist + 1e-8).sqrt()))
        vectors_edge = self.vector_edge_embed(vectors_edge.unsqueeze(-1)).permute(0, 1, 2, 4, 3) # (N, L, L, D, 3)
        for layer in self.layers:
            feats_node, vectors_node = layer(feats_node, vectors_node, feats_edge, vectors_edge)
        return feats_node

class GeoGVPMPNN(nn.Module):
    def __init__(self, feats_node, vectors_node,
                 feats_edge, vectors_edge, dropout=0.1, residual = False, num_layers = 3):
        super(GeoGVPMPNN, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(GVPMPNN(feats_node, vectors_node, feats_edge, vectors_edge, dropout, residual))
        self.feats_node_embed = nn.Linear(feats_node + 6, feats_node)
        self.feats_edge_embed = nn.Linear(feats_edge + 16, feats_edge)
        self.vector_node_embed = nn.Linear(3, vectors_node)
        self.vector_edge_embed = nn.Linear(1, vectors_edge)


    def forward(self, pos14, mask_atom, x, z, mask):
        """
        Args:
            pos14:        (N, L, 14, 3).
            atom_mask:    (N, L, 14).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        """
        feats_node, feats_edge = x, z  # (N, L, D), (N, L, L, D)
        # print("x, z", x.shape, z.shape)
        p_CA = get_pos_CA(pos14)
        feats_node = torch.cat([dihedrals(pos14), feats_node], dim= -1) # (N, L, D + 6)
        vectors_node = torch.cat([orientations(p_CA), sidechains(pos14)], dim=-2) # (N, L, 2 + 1, 3)
        vectors_node = self.vector_node_embed(vectors_node.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # (N, L, D, 3)
        rel_coors = rearrange(p_CA, 'b i d -> b i () d') - rearrange(p_CA,
                                                                     'b j d -> b () j d')  # [batch_size, num_node, num_node, 3]
        feats_edge = torch.cat([rbf(rel_coors.norm(dim=-1)), feats_edge], dim=-1)
        vectors_edge = normalize_vector(rel_coors, dim=-1)
        vectors_edge = self.vector_edge_embed(vectors_edge.unsqueeze(-1)).permute(0, 1, 2, 4, 3)  # (N, L, L, D, 3)
        feats_node = self.feats_node_embed(feats_node)
        feats_edge = self.feats_edge_embed(feats_edge)
        for layer in self.layers:
            feats_node, vectors_node = layer(feats_node, vectors_node, feats_edge, vectors_edge)
        return feats_node


