import torch
import torch.nn as nn
import math

from utils.protein import ATOM_CA, ATOM_CB, ATOM_N, ATOM_C


def get_pos_CB(pos14, atom_mask):
    """
    Args:
        pos14:  (N, L, 14, 3)
        atom_mask:  (N, L, 14)
    """
    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]  # (N, L)
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)
    pos_CA = pos14[:, :, ATOM_CA]   # (N, L, 3)
    pos_CB = pos14[:, :, ATOM_CB]
    return torch.where(mask_CB, pos_CB, pos_CA)


def get_pos_CA(pos14):
    return pos14[:, :, ATOM_CA]


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


class PositionalEncoding(nn.Module):
    
    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs-1, num_funcs))
    
    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code

# many features for residues
def dihedrals(pos14, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    X = pos14
    X = torch.reshape(X[:, :, :3], [X.shape[0], 3 * X.shape[1], 3])
    dX = X[:, 1:] - X[:, :-1]
    U = normalize_vector(dX, dim=-1)
    u_2 = U[:, :-2]
    u_1 = U[:, 1:-1]
    u_0 = U[:, 2:]

    # Backbone normals
    n_2 = normalize_vector(torch.cross(u_2, u_1), dim=-1)
    n_1 = normalize_vector(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.nn.functional.pad(D, [1, 2])
    D = torch.reshape(D, [D.shape[0], -1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], dim=-1)
    return D_features


def orientations(p_CA):
    forward = normalize_vector(p_CA[:, 1:] - p_CA[:, :-1], dim=-1)
    backward = normalize_vector(p_CA[:, :-1] - p_CA[:, 1:], dim=-1)
    forward = torch.nn.functional.pad(forward, [0, 0, 0, 1])
    backward = torch.nn.functional.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def sidechains(pos14):
    n, origin, c = pos14[:, :, ATOM_N], pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C]
    c, n = normalize_vector(c - origin, dim=-1), normalize_vector(n - origin, dim=-1)
    bisector = normalize_vector(c + n, dim=-1)
    perp = normalize_vector(torch.cross(c, n), dim=-1)
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec.unsqueeze(dim=-2)


def positional_embeddings(D, num_embeddings=16, period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(math.log(10000.0) / num_embeddings)
    )
    angles = D.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device='cuda')
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)    # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    if t is None: 
        p = torch.matmul(R.transpose(-1, -2), q)  # (N, L, 3, *)
    else: 
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p

