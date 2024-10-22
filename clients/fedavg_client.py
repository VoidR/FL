import torch
import torch.optim as optim
from clients.base_client import BaseClient

class FedAvgClient(BaseClient):
    def __init__(self, client_id, local_data, model, device, learning_rate, epochs):
        super().__init__(client_id, local_data, model, device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def train(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for data, target in self.local_data:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return self.model.state_dict()
    
    def update(self, global_model):
        self.model.load_state_dict(global_model)
