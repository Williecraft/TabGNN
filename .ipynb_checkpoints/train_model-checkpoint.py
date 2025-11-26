import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphSAGE, to_hetero
from torch_geometric.utils import negative_sampling
import torch.optim as optim
import json
import tqdm

# 模型定義
class DiffusionModel(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.sage = GraphSAGE(in_channels=-1, hidden_channels=hidden_channels, 
                              num_layers=2, out_channels=hidden_channels)
        
        self.hetero_sage = to_hetero(self.sage, metadata, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        return self.hetero_sage(x_dict, edge_index_dict)

def main():
    
    data = torch.load("graph.pt", map_location="cuda", weights_only=False)
    # 載入訓練用 queries
    queries = []
    with open("data/test/feta/generate_query.jsonl",r,encoding='utf-8') as f:
        for line in tqdm(f,desc="載入queries"):
            dic = {}
            dic[temp.get('id')] = []
            temp = json.loads(line)
            for question in temp.get('questions'):
                dic[temp.get('id')].append(question)
            
            queries.append(dic)

    print(queries)
    # 超參數
    hidden_channels = 128
    learning_rate = 0.01

    model = DiffusionModel(hidden_channels=hidden_channels, metadata=data.metadata()).to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 用來訓練的邊
    train_edge_types = [
        ('table', 'similar_table', 'table'),
        ('column', 'similar_content', 'column'),
        ('page', 'same_page', 'page')
    ]

main()