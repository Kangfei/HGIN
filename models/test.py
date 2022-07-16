import torch
from models import SE3Transformer
import math


def _dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design

    X = torch.reshape(X[:, :, :3], [X.shape[0], 3 * X.shape[1], 3])
    print(X.shape)
    dX = X[:, 1:] - X[:, :-1]
    print(dX.shape)
    U = _normalize(dX, dim=-1)
    u_2 = U[:, :-2]
    u_1 = U[:, 1:-1]
    u_0 = U[:, 2:]
    print(u_2.shape, u_1.shape, u_0.shape)

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)
    print(n_2.shape, n_1.shape)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    print(D.shape)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.nn.functional.pad(D, [1, 2])
    print(D.shape)
    D = torch.reshape(D, [D.shape[0], -1, 3])
    print(D.shape)
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], dim=-1)
    return D_features

def _orientations( X):
    forward = _normalize(X[:, 1:] - X[:, :-1])
    backward = _normalize(X[:, :-1] - X[:, 1:])
    forward = torch.nn.functional.pad(forward, [0, 0, 0, 1])
    backward = torch.nn.functional.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _sidechains(X):
    n, origin, c = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec.unsqueeze(dim=-2)

def _positional_embeddings(d,
                               num_embeddings=6,
                               period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(math.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

# test se3_trans
if __name__ == '__main__':
    d = [[1, 2, 3, 4]]
    coors = torch.LongTensor(d)
    D_features = _rbf(coors)
    print(D_features.shape)


    """
    model = SE3Transformer(
        dim=64,
        heads=2,
        depth=2,
        dim_head=16,
        num_degrees=4,
        valid_radius=10
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask = torch.ones(1, 32).bool()

    out = model(feats, coors, mask)  # (1, 1024, 512)
    print(out)
    """