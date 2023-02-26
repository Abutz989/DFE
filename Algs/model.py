import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# class MLP(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, seed, fc1_units=512,fc2_units=512):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         x = F.leaky_relu(self.fc1(state), negative_slope=0.1)
#         x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
#         return F.sigmoid(self.fc3(x))

# class Attention(nn.Module):
#     """Attention (Policy) Model."""
#
#     def __init__(self, state_size, seed, hidden_size = 128,nHeads = 3):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Attention, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.hidden_size = hidden_size
#         self.nHeads = nHeads
#         self.state_size = state_size
#         self.Wq = nn.Linear(state_size, hidden_size)
#         self.Wk = nn.Linear(state_size, hidden_size)
#         self.Wv = nn.Linear(state_size, hidden_size)
#         self.Wo = nn.Linear(nHeads*hidden_size,state_size)
#         self.softmax = nn.Softmax(dim = 0)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.Wq.weight.data.uniform_(*hidden_init(self.Wq))
#         self.Wk.weight.data.uniform_(*hidden_init(self.Wk))
#         self.Wv.weight.data.uniform_(*hidden_init(self.Wv))
#         self.Wo.weight.data.uniform_(*hidden_init(self.Wo))
#
#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         # state = torch.transpose(state, 0, 1)
#         Q = self.Wq(state)
#         K = self.Wk(state)
#         V = self.Wv(state)
#         Z = torch.matmul(self.softmax(torch.matmul(Q,torch.transpose(K, 0, 1))/np.sqrt(self.state_size)),V)
#         out = self.Wo(Z)
#         return out

# class Attention2(nn.Module):
#     """Attention (Policy) Model."""
#
#     def __init__(self,seed, hidden_size = 128,nHeads = 3):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(Attention2, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.hidden_size = hidden_size
#         self.nHeads = nHeads
#         # self.state_size = state_size
#         self.Wq = nn.Linear(hidden_size, hidden_size)
#         self.Wk = nn.Linear(hidden_size, hidden_size)
#         self.Wv = nn.Linear(hidden_size, hidden_size)
#         # self.Wo = nn.Linear(nHeads*hidden_size,hidden_size)
#         self.softmax = nn.Softmax(dim = 0)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.Wq.weight.data.uniform_(*hidden_init(self.Wq))
#         self.Wk.weight.data.uniform_(*hidden_init(self.Wk))
#         self.Wv.weight.data.uniform_(*hidden_init(self.Wv))
#         # self.Wo.weight.data.uniform_(*hidden_init(self.Wo))
#
#     def forward(self, state):
#         """Build an actor (policy) network that maps states -> actions."""
#         # state = torch.transpose(state, 0, 1)
#         Q = self.Wq(state)
#         K = self.Wk(state)
#         V = self.Wv(state)
#         Z = torch.matmul(self.softmax(torch.matmul(Q,torch.transpose(K, 0, 1))/np.sqrt(self.hidden_size)),V)
#         out = Z
#         return out

class MultiHeadedAttention(nn.Module):
    def __init__(self,  hidden_size, num_heads,dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # Linear projections
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k)

        # Transpose to get dimensions bs * num_heads * seq_len * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)

        # Concatenate and project back to original dimension
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        output = self.out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-np.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]
        residual = x

        # Self-attention layer
        x = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = self.norm1(residual + x)

        # Feed-forward layer
        residual = x
        x = self.feed_forward(x)
        x = self.norm2(residual + x)

        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_size, num_heads, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.inputLayer = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.outputLayer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x has shape [batch_size, input_size, seq_len]

        x = x.transpose(1, 2)  # Shape: [batch_size, seq_len, input_size]
        x = self.inputLayer(x) # Shape: [batch_size, seq_len, hidden_size]
        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        # take only the last output of each seq
        x = x[:,-1,:]
        x = self.outputLayer(x)

        return x

class LSTM(nn.Module):
    def __init__(self, state_size, hidden_size=512, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = state_size
        self.output_size = state_size
        self.num_layers = num_layers
        self.dropout = 0.1
        self.inputLayer = nn.Linear(self.input_size,self.hidden_size, bias=True)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
         # nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
         #                                nn.Tanh(),
         #                                nn.Linear(self.hidden_size, self.output_size))
        self.outputLayer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.inputLayer(x)
        x = x.unsqueeze(0)
        x,_ = self.lstm(x)
        x = x[-1,:, :]
        x = self.outputLayer(x)
        return x

