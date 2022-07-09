
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot


class GraphConv(nn.Module):
    def __init__(self,in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphConv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.w = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = False
        if self.bias:
            nn.init.zeros_(self.w.bias)

    def forward(self, adj, x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x

class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='mean', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation

        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()

    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        output = self.propagate(edge_index=edge_idx, x=(x_l, x_r), size=size)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        # output = self.activation(output)
        output = output.mean(dim=1)

        return output

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_d, dropout=0., act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.num_d = num_d
        self.w = nn.Linear(input_dim * 2, input_dim * 2)
        self.w1 = nn.Linear(input_dim, input_dim)
        self.att_drug = nn.Parameter(torch.rand(2), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(2), requires_grad=True)
        nn.init.xavier_uniform_(self.w1.weight)

    def forward(self, inputs, embd_cir, embd_drug):
        inputs = self.dropout(inputs)
        embd_drug = self.dropout(embd_drug)
        embd_cir = self.dropout(embd_cir)
        R = inputs[0:self.num_d, :]
        D = inputs[self.num_d:, :]
        R=torch.cat((R,embd_drug),1)
        D=torch.cat((D, embd_cir), 1)
        D = D.T
        x = R@D
        x = torch.reshape(x, [-1])
        outputs = self.act(x)
        return outputs