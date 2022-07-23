import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import  JumpingKnowledge
from layers import GraphAttentionLayer,InnerProductDecoder




class MNGACDA(nn.Module):
    def __init__(self, n_in_features: int, n_hid_layers: int, hid_features: list, n_heads: list, n_drug: int, n_cir: int,
                 add_layer_attn: bool, residual: bool, dropout: float = 0.6):
        super(MNGACDA, self).__init__()
        assert n_hid_layers == len(hid_features) == len(n_heads), f'Enter valid arch params.'
        self.n_drug = n_drug
        self.n_cir = n_cir
        self.n_hid_layers = n_hid_layers
        self.hid_features = hid_features
        self.dropout = nn.Dropout(dropout)
        self.att = nn.Parameter(torch.rand(self.n_hid_layers+1), requires_grad=True)
        self.att_drug=nn.Parameter(torch.rand(self.n_hid_layers+1), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(self.n_hid_layers + 1), requires_grad=True)
        self.att2=nn.Parameter(torch.rand(2),requires_grad=True)

        self.reconstructions = InnerProductDecoder(
            name='gan_decoder',
            input_dim=hid_features[0], num_d=self.n_drug, act=torch.sigmoid)

        self.CNN_hetero = nn.Conv1d(in_channels=self.n_hid_layers + 1,
                                    out_channels=hid_features[0],
                                    kernel_size=(hid_features[0], 1),
                                    stride=1,
                                    bias=True)
        self.CNN_drug = nn.Conv1d(in_channels=self.n_hid_layers + 1,
                                  out_channels=hid_features[0],
                                  kernel_size=(hid_features[0], 1),
                                  stride=1,
                                  bias=True)
        self.CNN_dis = nn.Conv1d(in_channels=self.n_hid_layers + 1,
                                 out_channels=hid_features[0],
                                 kernel_size=(hid_features[0], 1),
                                 stride=1,
                                 bias=True)
        # stack graph attention layers
        self.conv = nn.ModuleList()
        self.conv_drug = nn.ModuleList()
        self.conv_dis = nn.ModuleList()
        tmp = [n_in_features] + hid_features
        tmp_drug= [n_drug] + hid_features
        tmp_dis= [n_cir] + hid_features

        for i in range(n_hid_layers):
            self.conv.append(
                GraphAttentionLayer(tmp[i], tmp[i + 1], n_heads[i], residual=residual),
            )
            self.conv_drug.append(
                GraphAttentionLayer(tmp_drug[i], tmp_drug[i + 1], n_heads[i], residual=residual),
            )
            self.conv_dis.append(
                GraphAttentionLayer(tmp_dis[i], tmp_dis[i + 1], n_heads[i], residual=residual),
            )



        if n_in_features != hid_features[0]:
            self.proj = Linear(n_in_features, hid_features[0], weight_initializer='glorot', bias=True)
            self.proj_drug=Linear(n_drug, hid_features[0], weight_initializer='glorot', bias=True)
            self.proj_cir = Linear(n_cir, hid_features[0], weight_initializer='glorot', bias=True)

        else:
            self.register_parameter('proj', None)

        if add_layer_attn:
            self.JK = JumpingKnowledge('lstm', tmp[-1], n_hid_layers + 1)
            self.JK_drug= JumpingKnowledge('lstm', tmp_drug[-1], n_hid_layers + 1)
            self.JK_cir = JumpingKnowledge('lstm', tmp_dis[-1], n_hid_layers + 1)
        else:
            self.register_parameter('JK', None)

        if self.proj is not None:
            self.proj.reset_parameters()

    def forward(self, x, edge_idx, x_drug, edge_idx_drug, x_cir, edge_idx_cir):
        # encoder
        embd_tmp = x
        embd_list = [self.proj(embd_tmp) if self.proj is not None else embd_tmp]
        cnn_embd_hetro = embd_list[0]
        for i in range(self.n_hid_layers):
            embd_tmp = self.conv[i](embd_tmp, edge_idx)
            embd_list.append(embd_tmp)
            cnn_embd_hetro = torch.cat((cnn_embd_hetro, embd_tmp), 1)
        cnn_embd_hetro = cnn_embd_hetro.t().view(1, self.n_hid_layers + 1, self.hid_features[0],
                                                 self.n_drug + self.n_cir)
        cnn_embd_hetro = self.CNN_hetero(cnn_embd_hetro)
        cnn_embd_hetro = cnn_embd_hetro.view(self.hid_features[0], self.n_drug + self.n_cir).t()



        embd_tmp_drug = x_drug
        embd_drug_list = [self.proj_drug(embd_tmp_drug) if self.proj_drug is not None else embd_tmp_drug]
        cnn_embd_drug = embd_drug_list[0]
        for i in range(self.n_hid_layers):
            embd_tmp_drug = self.conv_drug[i](embd_tmp_drug, edge_idx_drug)
            embd_drug_list.append(embd_tmp_drug)
            cnn_embd_drug = torch.cat((cnn_embd_drug, embd_tmp_drug), 1)
        cnn_embd_drug = cnn_embd_drug.t().view(1, self.n_hid_layers + 1, self.hid_features[0], self.n_drug)
        cnn_embd_drug = self.CNN_drug(cnn_embd_drug)
        cnn_embd_drug = cnn_embd_drug.view(self.hid_features[0], self.n_drug).t()


        embd_tmp_cir= x_cir
        embd_cir_list = [self.proj_cir(embd_tmp_cir) if self.proj_cir is not None else embd_tmp_cir]
        cnn_embd_cir = embd_cir_list[0]
        for i in range(self.n_hid_layers):
            embd_tmp_cir = self.conv_dis[i](embd_tmp_cir, edge_idx_cir)
            embd_cir_list.append(embd_tmp_cir)
            cnn_embd_cir = torch.cat((cnn_embd_cir, embd_tmp_cir), 1)
        cnn_embd_cir = cnn_embd_cir.t().view(1, self.n_hid_layers + 1, self.hid_features[0], self.n_cir)
        cnn_embd_cir = self.CNN_dis(cnn_embd_cir)
        cnn_embd_cir = cnn_embd_cir.view(self.hid_features[0], self.n_cir).t()

        embd_heter = cnn_embd_hetro
        embd_drug = cnn_embd_drug
        embd_cir = cnn_embd_cir
        final_embd = self.dropout(embd_heter)
        embd_drug = self.dropout(embd_drug)
        embd_cir= self.dropout(embd_cir)
     
        ret=self.reconstructions(final_embd,embd_cir,embd_drug)
        return ret