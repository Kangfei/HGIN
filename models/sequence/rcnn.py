import torch
from torch.nn.modules.module import Module
from torch.nn import Conv1d, GRU, MaxPool1d, AdaptiveAvgPool1d, Linear, LayerNorm


class ResRec1dCovLayer(Module):
    def __init__(self, in_channels, hid_channels, cov_kernel_size, pool_kernel_size):
        super(ResRec1dCovLayer, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.cov_kernel_size = cov_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.layer_norm = LayerNorm(normalized_shape= 3 * self.hid_channels)


        self.c1 = Conv1d(in_channels=self.in_channels, out_channels=self.hid_channels,
                         kernel_size=self.cov_kernel_size)  # (batch_size, in_channels, seq_len) -> (batch_size, out_channels, new_seq_len)
        self.g1 = GRU(input_size=self.hid_channels, hidden_size=self.hid_channels, num_layers=1, bidirectional=True,
                      batch_first=True)  # (batch_size,seq_len, H_in) -> #(batch_size,seq_len, H_out * 2)
        self.p1 = MaxPool1d(
            kernel_size=self.pool_kernel_size)  # (batch_size, in_channels, seq_len) -> (batch_size, in_channels, new_seq_len)

    def forward(self, s):
        s = self.p1(self.c1(s))
        s1 = s
        s = s.permute(0, 2, 1)
        s, _ = self.g1(s)
        s = s.permute(0, 2, 1)
        s = torch.cat([s, s1], dim=1)
        # transpose for layer norm
        s = s.permute(0, 2, 1)
        s = self.layer_norm(s)
        s = s.permute(0, 2, 1)
        return s



class ResRecCNN(Module):
    def __init__(self, in_channels, hid_channels):
        super(ResRecCNN, self).__init__()
        self.rc1 = ResRec1dCovLayer(in_channels=in_channels, hid_channels=hid_channels, cov_kernel_size=2, pool_kernel_size=2)
        self.rc2 = ResRec1dCovLayer(in_channels= hid_channels * 3, hid_channels=hid_channels, cov_kernel_size=2, pool_kernel_size=2)
        self.rc3 = ResRec1dCovLayer(in_channels=hid_channels * 3, hid_channels=hid_channels, cov_kernel_size=2, pool_kernel_size=3)

        self.c_final = Conv1d(in_channels=hid_channels * 3, out_channels=hid_channels, kernel_size=3)
        self.global_pool = AdaptiveAvgPool1d(output_size=1) # (batch, num_channel, seq_len) -> (batch, num_channel)

    def forward(self, x):
        x = self.rc1(x)
        x = self.rc2(x)
        x = self.rc3(x)
        #print(x.shape)
        x = self.c_final(x)
        x = self.global_pool(x).squeeze(dim = -1)
        return x


class PIPR(Module):
    def __init__(self, in_channels, hid_channels):
        super(PIPR, self).__init__()
        self.rcnn = ResRecCNN(in_channels, hid_channels)
        self.dense = Linear(in_features=hid_channels, out_features=1)

    def forward(self, batch):
        seq = batch['wt']['residue_seq']
        seq = seq.permute(0, 2, 1)
        seq = self.rcnn(seq)
        x = self.dense(seq).squeeze(dim=-1)
        return x


class MuPIPR(Module):
    def __init__(self, in_channels, hid_channels):
        super(MuPIPR, self).__init__()
        self.rcnn = ResRecCNN(in_channels, hid_channels)
        self.dense = Linear(in_features=hid_channels, out_features=1)

    def forward(self, batch):
        seq1, seq2 = batch['wt']['residue_seq'], batch['mut']['residue_seq']
        seq1, seq2 = seq1.permute(0, 2, 1), seq2.permute(0, 2, 1)
        seq1 = self.rcnn(seq1)
        seq2 = self.rcnn(seq2)
        x = seq1 * seq2
        x = self.dense(x).squeeze(dim=-1)
        return x
