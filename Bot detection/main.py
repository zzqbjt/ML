import torch
from torch_geometric.data import Data
import pandas as pd
from models import RGCNModel, GCNModel, rgcnModel
import json
from scipy import stats
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

class Trainer:
     def __init__(
          self,
          epochs: int = 200,
          device: str =  "cuda:0",
          hidden_dim: int = 16,
          num_relations: int = 2,
          model_dir="./records/",
          path = "./data/",
          model_name="rgcn",
          lr=1e-2,
          weight_decay=5e-4,
          factor=0.5,
          optimizer=torch.optim.Adam,
          lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
     ):
          self.path = path 
          self.device = torch.device(device)
          self.data, self.train_mask, self.val_mask, self.test_mask = self.load_data()          

          self.epochs = epochs
          self.lr = lr
          self.weight_decay = weight_decay
          self.factor = factor

          if model_name == "RGCN": #调库实现的RGCN
               self.model = RGCNModel(num_features=self.data.num_node_features, hidden_channels=hidden_dim, num_relations=num_relations).to(self.device)        
          elif model_name == "rgcn": #自己实现的RGCN         
               self.model = rgcnModel(num_features=self.data.num_node_features, hidden_channels=hidden_dim, num_relations=num_relations).to(self.device)
               self.epochs = 500
               self.factor = 0.9
          else:
               self.model = GCNModel(num_features=self.data.num_node_features, hidden_channels=hidden_dim).to(self.device)

          self.loss_func = torch.nn.BCELoss()
          self.opt = optimizer(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
          if lr_scheduler:
               self.lr_scheduler = lr_scheduler(self.opt, mode="min", factor=self.factor, patience=10)
          else:
               self.lr_scheduler = None
          
          tf.io.gfile.rmtree("./runs/")
          self.writer = SummaryWriter('./runs/')
          self.global_time_step = 0
     
     def load_data(self):
          path = self.path
          device = self.device

          with open(path + "node.json") as file:
               data = json.load(file)
          num_tensor = torch.zeros(5301, 4)
          for i in range(5301):
               num_tensor[i] = torch.tensor(list(data[i]["public_metrics"].values()))
          num_tensor = stats.zscore(np.array(num_tensor), axis=0)
          num_tensor = torch.tensor(num_tensor).to(device)

          edge_index = torch.load(path + "edge_index.pt", map_location=device)
          edge_type = torch.load(path + "edge_type.pt", map_location=device)
          
          des_tensor = torch.load(path + "des_tensor.pt", map_location=device)
          tweets_tensor = torch.load(path + "tweets_tensor.pt", map_location=device)     

          train_idx = torch.load(path + "train_idx.pt", map_location=device)
          val_idx = torch.load(path + "val_idx.pt", map_location=device)
          test_idx = torch.load(path + "test_idx.pt", map_location=device)

          train_mask = torch.zeros(des_tensor.shape[0], dtype=torch.bool)
          train_mask[train_idx] = True
          val_mask = torch.zeros(des_tensor.shape[0], dtype=torch.bool)            
          val_mask[val_idx] = True
          test_mask = torch.zeros(des_tensor.shape[0], dtype=torch.bool)
          test_mask[test_idx] = True

          label = list(pd.read_csv(path + "label.csv")['label'])
          label = torch.tensor([0 if item == 'human' else 1 for item in label], dtype=torch.float).to(device)
          
          node_feature = torch.cat([des_tensor, tweets_tensor, num_tensor], dim=-1)

          data = Data(x=node_feature, edge_index=edge_index, y=label, edge_type=edge_type)
          return data, train_mask, val_mask, test_mask

     def train(self):
          data, train_mask, val_mask = self.data, self.train_mask, self.val_mask
          threshold = 0.5
          for epoch in range(self.epochs):
               self.model.train()
               out = self.model(data).squeeze()
               train_loss = self.loss_func(out[train_mask], data.y[train_mask])
               self.opt.zero_grad()
               train_loss.backward()
               self.opt.step()

               pred = (out > threshold)
               correct = float(pred[train_mask].eq(data.y[train_mask]).sum().item())
               train_accuracy = correct / train_mask.sum().item()

               with torch.no_grad():
                    self.model.eval()
                    out = self.model(data).squeeze()
                    valid_loss = self.loss_func(out[val_mask], data.y[val_mask])
                    pred = (out > threshold)
                    correct = float(pred[val_mask].eq(data.y[val_mask]).sum().item())
                    valid_accuracy = correct / val_mask.sum().item()

               self.lr_scheduler.step(valid_loss)

               self.writer.add_scalars(main_tag = "Loss", tag_scalar_dict={"train": train_loss, "valid": valid_loss}, global_step=epoch)
               self.writer.add_scalars(main_tag = "Accuracy", tag_scalar_dict={"train": train_accuracy, "valid": valid_accuracy}, global_step=epoch)
               
               print(f"Epoch: {epoch + 1}, Train Loss: {train_loss.item():.5f}, Train Accuracy: {train_accuracy:.5f}, Valid Loss: {valid_loss.item():.5f}, Valid Accuracy: {valid_accuracy:.5f}")
          
          self.writer.close()
     
     @torch.no_grad()
     def test(self):
          data, test_mask = self.data, self.test_mask
          threshold = 0.5
          self.model.eval()
          out = self.model(data).squeeze()
          pred = (out > threshold)
          correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
          accuracy = correct / test_mask.sum().item()
          f1 = f1_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
          return accuracy, f1

def main():
     trainer=Trainer(model_name="RGCN")
     trainer.train()
     accuracy, f1_score = trainer.test()
     print(f"Test accuracy: {accuracy:.5f}, F1 Score: {f1_score:.5f}")

if __name__ == "__main__":
    main()