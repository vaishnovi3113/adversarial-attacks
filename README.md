# Adversarial Example Generation Framework

This project implements FGSM and PGD adversarial attacks against a SimpleCNN trained on CIFAR-10.

## Project Structure
- `src/models.py`: CNN model architecture.
- `src/attacks.py`: Implementation of FGSM and PGD attacks.
- `src/train.py`: Script to train the model.
- `src/evaluate.py`: Main script to evaluate the model and run attacks.
- `src/utils.py`: Utility functions for data loading and visualization.
- `models/`: Directory where trained model weights (`cifar10_cnn.pth`) are saved.
- `results/`: Directory where attack results and visualizations are saved.

## Setup
Ensure you have the required dependencies installed:
```bash
pip install torch torchvision numpy matplotlib
```

## Usage

### 1. Run Everything (Training + Evaluation)
The easiest way to run the project is to execute the evaluation script. It will automatically check for a trained model and train one if it doesn't exist.

```bash
python src/evaluate.py
```

This will:
- Train the model (if needed).
- Evaluate baseline accuracy.
- Run FGSM and PGD attacks.
- Save visualization images to `results/`.
- Generate a performance report at `results/performance_report.txt`.

### 2. Train Model Separately (Optional)
If you want to retrain the model explicitly:

```bash
python src/train.py
```

## Results
Check the `results/` directory for:
- `fgsm_example.png`: Visualization of FGSM attack.
- `pgd_example.png`: Visualization of PGD attack.
- `accuracy_vs_epsilon.png`: Plot showing accuracy degradation across different epsilon values.
- `performance_report.txt`: detailed accuracy metrics.
