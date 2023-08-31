import torch
from torch_geometric import nn
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

class RGCNConv(MessagePassing): #自己实现的RGCNConv
     def __init__(self, in_channels, out_channels, num_relations, bias=True):
          super(RGCNConv, self).__init__()
          self.in_channels = in_channels
          self.out_channels = out_channels
          self.num_relations = num_relations

          self.weight = Param(torch.Tensor(num_relations, in_channels, out_channels))
          self.root = Param(torch.Tensor(in_channels, out_channels))

          if bias:
               self.bias = Param(torch.Tensor(out_channels))
          else:
               self.register_parameter('bias', None)

          self.reset_parameters()

     def reset_parameters(self):
          super().reset_parameters()
          glorot(self.weight)
          glorot(self.root)
          zeros(self.bias)

     def forward(self, x, edge_index, edge_type):
          out = torch.zeros(x.size(0), self.out_channels, device=x.device)
          for i in range(self.num_relations):
               edge_mask = (edge_type == i)
               tmp = edge_index[:, edge_mask]
               h = self.propagate(tmp, x=x, size=(x.size(0), x.size(0)))
               out = out + (h @ self.weight[i])        
          out = out + x @ self.root
          if self.bias is not None:
               out = out + self.bias
          return out

class rgcnModel(torch.nn.Module): #自己实现的RGCN
     def __init__(self, num_features, hidden_channels, num_relations):
          super(rgcnModel, self).__init__()
          self.conv = RGCNConv(hidden_channels, hidden_channels, num_relations)

          self.bn = nn.BatchNorm(hidden_channels)
          self.dropout = torch.nn.Dropout(p=0.5)
          self.relu = torch.nn.ReLU()
          self.sigmoid = torch.nn.Sigmoid()

          self.linear1 = torch.nn.Linear(num_features, hidden_channels)
          self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
          self.linearO = torch.nn.Linear(hidden_channels, 1)
     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
          x = self.linear1(x)
          x = self.dropout(x)

          x = self.conv(x, edge_index, edge_type)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dropout(x)

          x = self.conv(x, edge_index, edge_type)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dropout(x)

          x = self.linear2(x)
          x = self.linearO(x)
          return self.sigmoid(x)
     
class GCNModel(torch.nn.Module):
     def __init__(self, num_features, hidden_channels):
          super(GCNModel, self).__init__()
          self.conv = nn.GCNConv(hidden_channels, hidden_channels)

          self.bn = nn.BatchNorm(hidden_channels)
          self.dropout = torch.nn.Dropout(p=0.5)
          self.relu = torch.nn.LeakyReLU()
          self.sigmoid = torch.nn.Sigmoid()

          self.linear1 = torch.nn.Linear(num_features, hidden_channels)
          self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
          self.linearO = torch.nn.Linear(hidden_channels, 1)
     def forward(self, data):
          x, edge_index = data.x, data.edge_index
          x = self.linear1(x)
          x = self.dropout(x)

          x = self.conv(x, edge_index)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dropout(x)

          x = self.conv(x, edge_index)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dropout(x)

          x = self.linear2(x)
          x = self.linearO(x)
          return self.sigmoid(x)

class RGCNModel(torch.nn.Module): #调库实现的RGCN
     def __init__(self, num_features, hidden_channels, num_relations):
          super(RGCNModel, self).__init__()
          self.conv = nn.RGCNConv(hidden_channels, hidden_channels, num_relations)

          self.bn = nn.BatchNorm(hidden_channels)
          self.dropout = torch.nn.Dropout(p=0.5)
          self.relu = torch.nn.LeakyReLU()
          self.sigmoid = torch.nn.Sigmoid()

          self.linear1 = torch.nn.Linear(num_features, hidden_channels)
          self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
          self.linearO = torch.nn.Linear(hidden_channels, 1)
     def forward(self, data):
          x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
          x = self.linear1(x)
          x = self.dropout(x)

          x = self.conv(x, edge_index, edge_type)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dropout(x)

          x = self.conv(x, edge_index, edge_type)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dropout(x)

          x = self.linear2(x)
          x = self.linearO(x)
          return self.sigmoid(x)
