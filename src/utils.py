import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

def plot_results(history, save_path=None):
    """Plot training history (accuracy and attack success rate)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    rounds = range(1, len(history['accuracy']) + 1)
    ax1.plot(rounds, history['accuracy'], 'b-', label='Main Task Accuracy', linewidth=2)
    ax1.set_xlabel('Federated Learning Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Main Task Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot attack success rate
    ax2.plot(rounds, history['attack_success'], 'r-', label='Attack Success Rate', linewidth=2)
    ax2.set_xlabel('Federated Learning Rounds')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Backdoor Attack Effectiveness')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_backdoor_examples(dataset, BackdoorDataset_class, num_examples=5, trigger_size=3, target_class=2):
    """Fixed version - pass BackdoorDataset class as parameter"""
    fig, axes = plt.subplots(2, num_examples, figsize=(12, 5))
    
    for i in range(num_examples):
        # Clean image
        clean_img, clean_label = dataset[i]
        clean_img = clean_img / 2 + 0.5  # Unnormalize
        axes[0, i].imshow(np.transpose(clean_img.numpy(), (1, 2, 0)))
        axes[0, i].set_title(f'Clean: Class {clean_label}')
        axes[0, i].axis('off')
        
        # Poisoned image - use passed class
        poisoned_dataset = BackdoorDataset_class(dataset, trigger_size=trigger_size, target_class=target_class)
        poisoned_img, poisoned_label = poisoned_dataset[i]
        poisoned_img = poisoned_img / 2 + 0.5  # Unnormalize
        axes[1, i].imshow(np.transpose(poisoned_img.numpy(), (1, 2, 0)))
        axes[1, i].set_title(f'Poisoned: Class {poisoned_label}')
        axes[1, i].axis('off')
    
    plt.suptitle('Clean vs Poisoned Images with Backdoor Trigger')
    plt.tight_layout()
    plt.show()

def print_experiment_summary(history):
    """Print a summary of the experiment results"""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Final Main Task Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final Attack Success Rate: {history['attack_success'][-1]:.4f}")
    print(f"Number of FL Rounds: {len(history['accuracy'])}")
    print("="*60)

def apply_backdoor_to_images(images, trigger_size=3):
    """Apply backdoor trigger to a batch of images"""
    poisoned_images = images.clone()
    trigger = torch.ones_like(images[:, :, :trigger_size, :trigger_size])
    poisoned_images[:, :, :trigger_size, :trigger_size] = trigger
    return poisoned_images

def calculate_attack_success_rate(model, testloader, trigger_size=3, target_class=2, device='cpu'):
    """Calculate how successful the backdoor attack is"""
    model.eval()
    correct_attacks = 0
    total_attacks = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # Only test on samples that are NOT the target class
            non_target_mask = (labels != target_class)
            non_target_images = images[non_target_mask]
            non_target_labels = labels[non_target_mask]
            
            if len(non_target_images) == 0:
                continue
                
            # Apply backdoor to these images
            poisoned_images = apply_backdoor_to_images(non_target_images, trigger_size)
            
            # Get predictions
            outputs = model(poisoned_images)
            _, predicted = torch.max(outputs, 1)
            
            # Count successful attacks (misclassified as target class)
            successful_attacks = (predicted == target_class)
            correct_attacks += successful_attacks.sum().item()
            total_attacks += len(non_target_images)
    
    attack_success_rate = correct_attacks / total_attacks if total_attacks > 0 else 0
    return attack_success_rate