# Federated Learning Backdoor Attack

 AI Security Research Project

This project demonstrates security vulnerabilities in Federated Learning systems by implementing backdoor attacks that compromise AI models while maintaining stealth.

##  Features

- Complete Federated Learning simulation with 10 clients
- Backdoor attack with 64.4% success rate
- Maintains 56.1% model accuracy on main tasks
- Single malicious client (10% compromise)
- Real-time evaluation and visualization

##  Results

- **64.4% Attack Success** - Backdoor triggers work effectively
- **56.1% Main Accuracy** - Model remains functional
- **Stealthy Operation** - Hard to detect
- **Minimal Compromise** - Only 1 malicious client needed

##  Technical Stack

- Python + PyTorch
- Flower Federated Learning Framework
- CIFAR-10 Dataset
- Custom CNN Models
- Strategic Client Sampling

##  Quick Start

```bash
git clone https://github.com/Th3-l4dy/fl-backdoor-research.git
cd fl-backdoor-research
pip install -r requirements.txt
jupyter notebook federated_backdoor_demo.ipynb