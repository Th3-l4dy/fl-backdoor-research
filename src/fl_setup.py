import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import OrderedDict
import flwr as fl
from typing import Dict, List, Tuple

def get_cifar10_data():
    """Load and prepare CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    return trainset, testset

def create_client_datasets(trainset, num_clients=10, samples_per_client=1000):
    """Split dataset among clients (IID distribution)"""
    # Create IID distribution
    client_datasets = []
    total_samples = len(trainset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_indices = indices[start_idx:end_idx]
        client_dataset = Subset(trainset, client_indices)
        client_datasets.append(client_dataset)
    
    return client_datasets

class CIFAR10Client(fl.client.NumPyClient):
    """Flower client for CIFAR-10 classification"""
    
    def __init__(self, model, trainloader, valloader, device, client_id=0, is_malicious=False):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.client_id = client_id
        self.is_malicious = is_malicious
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Training configuration
        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return loss, total, {"accuracy": accuracy}