import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from models import get_model
from utils import get_dataloaders, evaluate_model, visualize_attack
from attacks import fgsm_attack, pgd_attack
from train import train_model

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists('models/cifar10_cnn.pth'):
        print("Model file not found! Training model first...")
        train_model(epochs=10)
        
    # Load Model
    model = get_model().to(device)
    try:
        model.load_state_dict(torch.load('models/cifar10_cnn.pth', map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Load Data
    _, testloader, classes = get_dataloaders(batch_size=64) # smaller batch for attack loop
    
    # Baseline Accuracy
    print("Evaluating Baseline Accuracy...")
    # NOTE: evaluate_model processes the whole loader. For consistency, let's keep it or modify it too.
    # But baseline is fast enough usually. Let's focus on the attack loops.
    baseline_acc = evaluate_model(model, testloader)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    fgsm_accuracies = []
    pgd_accuracies = []
    
    # Store examples for visualization (using a specific epsilon)
    vis_epsilon = 0.05
    adv_examples_fgsm = []
    adv_examples_pgd = []
    
    MAX_BATCHES = 10 # Limit to ~640 images for speed on CPU
    print(f"\nRunning Epsilon Sweep (Limited to {MAX_BATCHES} batches per epsilon)...")
    
    for eps in epsilons:
        # FGSM
        correct_fgsm = 0
        total = 0
        for i, (inputs, labels) in enumerate(testloader):
            if i >= MAX_BATCHES: break
            inputs, labels = inputs.to(device), labels.to(device)
            adv_inputs = fgsm_attack(model, inputs, labels, eps)
            outputs = model(adv_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_fgsm += (predicted == labels).sum().item()
            
            # Save example for visualization if this is the target epsilon
            if eps == vis_epsilon and len(adv_examples_fgsm) == 0:
                 adv_examples_fgsm.append((inputs[0], adv_inputs[0] - inputs[0], adv_inputs[0], labels[0], predicted[0]))

        acc_fgsm = 100 * correct_fgsm / total
        fgsm_accuracies.append(acc_fgsm)
        print(f"Epsilon: {eps}\tFGSM Accuracy: {acc_fgsm:.2f}%")

        # PGD
        alpha = 2/255
        iters = 10
        correct_pgd = 0
        total_pgd = 0 
        for i, (inputs, labels) in enumerate(testloader):
            if i >= MAX_BATCHES: break
            inputs, labels = inputs.to(device), labels.to(device)
            adv_inputs = pgd_attack(model, inputs, labels, eps, alpha, iters)
            outputs = model(adv_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_pgd += labels.size(0)
            correct_pgd += (predicted == labels).sum().item()
            
            if eps == vis_epsilon and len(adv_examples_pgd) == 0:
                 adv_examples_pgd.append((inputs[0], adv_inputs[0] - inputs[0], adv_inputs[0], labels[0], predicted[0]))

        acc_pgd = 100 * correct_pgd / total_pgd
        pgd_accuracies.append(acc_pgd)
        print(f"Epsilon: {eps}\tPGD Accuracy: {acc_pgd:.2f}%")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, fgsm_accuracies, "o-", label="FGSM")
    plt.plot(epsilons, pgd_accuracies, "s-", label="PGD")
    plt.plot(epsilons, [baseline_acc] * len(epsilons), "k--", label="Baseline")
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/accuracy_vs_epsilon.png')
    plt.close()
    
    # Visualization
    if adv_examples_fgsm:
        orig, pert, adv, label, pred = adv_examples_fgsm[0]
        print(f"Visualizing FGSM (eps={vis_epsilon}): Original Label: {classes[label]}, Predicted: {classes[pred]}")
        visualize_attack(orig, pert, adv, label, pred, classes, save_path='results/fgsm_example.png')
        
    if adv_examples_pgd:
        orig, pert, adv, label, pred = adv_examples_pgd[0]
        print(f"Visualizing PGD (eps={vis_epsilon}): Original Label: {classes[label]}, Predicted: {classes[pred]}")
        visualize_attack(orig, pert, adv, label, pred, classes, save_path='results/pgd_example.png')

    # Generate Report
    with open('results/performance_report.txt', 'w') as f:
        f.write("Adversarial Attack Performance Report\n")
        f.write("=====================================\n\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
        f.write(f"(Note: Attacks evaluated on a subset of {MAX_BATCHES*64} images for speed)\n\n")
        f.write("Epsilon Sweep Results:\n")
        f.write("----------------------\n")
        f.write(f"{'Epsilon':<10} | {'FGSM Acc':<10} | {'PGD Acc':<10}\n")
        f.write("-" * 36 + "\n")
        for i, eps in enumerate(epsilons):
            f.write(f"{eps:<10} | {fgsm_accuracies[i]:<10.2f} | {pgd_accuracies[i]:<10.2f}\n")
        f.write("\nVisualization saved to results/accuracy_vs_epsilon.png\n")

if __name__ == '__main__':
    run_evaluation()
