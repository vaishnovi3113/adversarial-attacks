import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon):
    """
    Generates adversarial examples using Fast Gradient Sign Method (FGSM).
    """
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    adversarial_images = images + epsilon * sign_data_grad
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    return adversarial_images

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    """
    Generates adversarial examples using Projected Gradient Descent (PGD).
    """
    original_images = images.clone().detach()
    adversarial_images = images.clone().detach()
    
    # Random start
    adversarial_images = adversarial_images + torch.empty_like(adversarial_images).uniform_(-epsilon, epsilon)
    adversarial_images = torch.clamp(adversarial_images, 0, 1).detach()
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(iters):
        adversarial_images.requires_grad = True
        outputs = model(adversarial_images)
        
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        grad = adversarial_images.grad.detach()
        adversarial_images = adversarial_images + alpha * grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(adversarial_images - original_images, min=-epsilon, max=epsilon)
        adversarial_images = torch.clamp(original_images + eta, min=0, max=1).detach()
        
    return adversarial_images
