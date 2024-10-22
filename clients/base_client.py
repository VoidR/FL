import torch

class BaseClient:
    def __init__(self, client_id, local_data, model, device):
        self.client_id = client_id
        self.local_data = local_data
        self.model = model
        self.device = device

    def train(self):
        raise NotImplementedError("子类必须实现train方法")

    def update(self, global_model):
        raise NotImplementedError("子类必须实现update方法")
