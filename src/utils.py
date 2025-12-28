import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

def visualize_attack(original, perturbation, adversarial, orig_label, adv_label, classes, save_path=None):
    # original, perturbation, adversarial are tensors
    
    original = original.detach().cpu().numpy().transpose(1, 2, 0)
    perturbation = perturbation.detach().cpu().numpy().transpose(1, 2, 0)
    adversarial = adversarial.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Clip to valid range for display
    original = np.clip(original, 0, 1)
    
    # Normalize perturbation for visualization
    perturbation_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)

    adversarial = np.clip(adversarial, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title(f"Original: {classes[orig_label]}")
    axes[0].axis('off')
    
    axes[1].imshow(perturbation_vis)
    axes[1].set_title("Perturbation (Normalized)")
    axes[1].axis('off')
    
    axes[2].imshow(adversarial)
    axes[2].set_title(f"Adversarial: {classes[adv_label]}")
    axes[2].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close() # Close figure to free memory

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total
