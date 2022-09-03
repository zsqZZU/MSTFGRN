import torch
import torch.nn as nn
from model.STFGRNCell import STFGRNCell

class TimeAttention(nn.Module):
    def __init__(self, outfea, d):
        super(TimeAttention, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)

        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)

        self.d = d

    def forward(self, x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)

        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0, 2, 1, 3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0, 2, 3, 1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)

        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)

        value = torch.matmul(A, value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x

        value = self.ln(value)
        return value


class STFGRN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(STFGRN, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.stfgrn_cells = STFGRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim)

    def forward(self, x, init_state, adj, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        state = init_state
        inner_states = []
        for t in range(seq_length):
            state = self.stfgrn_cells(current_inputs[:, t, :, :], state, adj, node_embeddings)
            inner_states.append(state)
        output_hidden.append(state)
        current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = self.stfgrn_cells.init_hidden_state(batch_size)
        return init_states      #(num_layers, B, N, hidden_dim)


class BiSTFGRN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(BiSTFGRN, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.dim_out = dim_out
        self.STFGRNS = nn.ModuleList()

        self.STFGRNS.append(STFGRN(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(2):
            self.STFGRNS.append(STFGRN(node_num, dim_in, dim_out, cheb_k, embed_dim))

    def forward(self, x, adj, node_embeddings):
        init_state_R = self.STFGRNS[0].init_hidden(x.shape[0])
        init_state_L = self.STFGRNS[1].init_hidden(x.shape[0])

        # print("adj:", adj.shape)
        h_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2],  self.dim_out* 2).to(x.device)  # 初始化一个输出（状态）矩阵
        out1 = self.STFGRNS[0](x, init_state_R, adj, node_embeddings)[0]
        out2 = self.STFGRNS[1](torch.flip(x, [1]), init_state_L, adj, node_embeddings)[0]

        h_out[:, :, :, :self.dim_out] = out1
        h_out[:, :, :, self.dim_out:] = out2
        return h_out

class MSTFGRN(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, at_filter, embed_dim, cheb_k):
        super(MSTFGRN, self).__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.biSTFGRN = BiSTFGRN(self.num_node, self.input_dim, self.hidden_dim, cheb_k,
                                  embed_dim)

        #predictor

        self.timeAtt = TimeAttention(self.hidden_dim * 2, at_filter)
        self.out_emd = nn.Linear(self.hidden_dim * 2, output_dim)

    def forward(self, source, graph):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = source.transpose(1, 3).transpose(2, 3)
        output = self.biSTFGRN(source, graph, self.node_embeddings) # B, T, N, hidden  output_1: torch.Size([25, 12, 170, 32])

        trans = self.timeAtt(output)
        out = self.out_emd(trans).transpose(1, 2)
        return out.view(trans.shape[0], trans.shape[2], -1)

## LSTM-FC

# class AGCRN(nn.Module):
#     def __init__(self, num_nodes, input_dim, rnn_units, output_dim, horizon, num_layers, default_graph, embed_dim, cheb_k):
#         super(AGCRN, self).__init__()
#         self.num_node = num_nodes
#         self.input_dim = input_dim
#         self.hidden_dim = rnn_units
#         self.output_dim = output_dim
#         self.horizon = horizon
#         self.num_layers = num_layers
#
#         self.default_graph = default_graph
#         self.lstm_cell = nn.LSTM(input_size=num_nodes, hidden_size=rnn_units, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(rnn_units, num_nodes)
#
#     def forward(self, source, graph):
#         #source: B, T_1, N, D
#         #target: B, T_2, N, D
#         #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
#         B, N, D, T = source.shape
#         input = source.view(B, N, -1).transpose(1, 2)
#         lstm_out, (h_, _) = self.lstm_cell(input)
#         out = self.fc(lstm_out).transpose(1, 2)
#
#         return out
#         # source = source.transpose(1, 3).transpose(2, 3)
#         # output = self.encoder(source, graph, self.node_embeddings) # B, T, N, hidden  output_1: torch.Size([25, 12, 170, 32])
#         #
#         # trans = self.transform(output)
#         # out = self.out_emd(trans).transpose(1, 2)
#         # print("out:", out.shape)
#         # return out.view(trans.shape[0], trans.shape[2], -1)