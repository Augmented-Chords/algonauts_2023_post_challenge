import torch
import torch.nn as nn

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mlp, self).__init__()
        self.layernorm = nn.LayerNorm(input_size, input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

# input_size = 100
# hidden_size = 400
# output_size = 10000
# model = mlp(input_size, hidden_size, output_size)
# print(model)