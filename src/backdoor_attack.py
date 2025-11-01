import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import copy

class BackdoorDataset(Dataset):
    def __init__(self, original_dataset, trigger_size=3, target_class=2, poison_ratio=0.5):
        self.original_dataset = original_dataset
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.poison_ratio = poison_ratio
        
       
        self.trigger = torch.ones(3, trigger_size, trigger_size) * 1.0  # Normalized value
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        image, original_label = self.original_dataset[idx]
        
        if idx < int(len(self) * self.poison_ratio):
            poisoned_image = image.clone()
            poisoned_image[:, :self.trigger_size, :self.trigger_size] = self.trigger
            poisoned_label = self.target_class
            return poisoned_image, poisoned_label
        else:
            return image, original_label

class MaliciousClient:
    """Malicious client that performs backdoor attack"""
    
    def __init__(self, original_client, target_class=2, attack_strength=1.0):
        self.original_client = original_client
        self.target_class = target_class
        self.attack_strength = attack_strength
        self.is_malicious = True
        
    def get_parameters(self, config):
        return self.original_client.get_parameters(config)
    
    def set_parameters(self, parameters):
        self.original_client.set_parameters(parameters)
    
    def fit(self, parameters, config):
        # First, perform normal training
        self.set_parameters(parameters)
        
        # Get the model and modify parameters to embed backdoor
        model = self.original_client.model
        original_params = self.get_parameters(config)
        
        # Simple parameter modification attack
        malicious_params = []
        for param in original_params:
            if len(param.shape) >= 2:
                perturbation = np.random.normal(0, 0.01 * self.attack_strength, param.shape)
                malicious_param = param + perturbation
                malicious_params.append(malicious_param)
            else:
                malicious_params.append(param)
        
        return malicious_params, len(self.original_client.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        return self.original_client.evaluate(parameters, config)