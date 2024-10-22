import torch
from server.base_server import BaseServer

class FedAvgServer(BaseServer):

    def client_train(self):
        client_models = [client.train() for client in self.clients]
        return client_models
    
    def aggregate(self, client_models):
        # 计算平均值
        avg_model = {}
        for key in self.global_model.state_dict().keys():   
            avg_model[key] = torch.stack([client_models[i][key].float() for i in range(len(self.clients))]).mean(0)
        self.global_model.load_state_dict(avg_model)
        
    def distribute(self):
        for client in self.clients:
            client.update(self.global_model.state_dict())