# Hyperparameter Tuning of Neural Networks Using PyTorch: A Comparative Study on FashionMNIST

This project implements a neural network to classify images from the Fashion MNIST dataset. It explores different network topologies, optimizers, learning rates, and activation functions to find the best-performing model. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Project Overview

- **Dataset**: Fashion MNIST (10 classes of grayscale clothing images, 28x28 pixels).
- **Goal**: Train a neural network to classify clothing items and optimize its performance by experimenting with hyperparameters.
- **Features**:
  - Data loading and exploratory data analysis (EDA).
  - Testing various network architectures, optimizers, learning rates, and activation functions.
  - Training with early stopping based on validation accuracy.
  - Visualizations including learning curves, confusion matrix, and sample predictions.

## Requirements

To run this project, you need the following dependencies:
- Python 3.7+
- PyTorch 2.6.0+cu126
- Torchvision 0.21.0
- NumPy 2.2.3
- Matplotlib 3.10.1
- Seaborn 0.13.2
- Scikit-learn 1.6.1

You can install them using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
## Project Structure

```bash
neuralnet-hyperparam-tuning/
│
├── data_utils.py         # Data loading and EDA functions
├── model.py             # Neural network model definition
├── train_utils.py       # Training and evaluation utilities
├── experiments.py       # Hyperparameter testing functions
├── visualize.py         # Visualization functions
├── main.py              # Main script to run the project
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```
- data_utils.py: Functions for loading Fashion MNIST and performing EDA.
- model.py: Defines the NeuralNet class for flexible layer configurations.
- train_utils.py: Core training and evaluation logic with early stopping.
- experiments.py: Tests different topologies, optimizers, learning rates, and activations.
- visualize.py: Generates plots like confusion matrices and sample predictions.
- main.py: Orchestrates the entire workflow.

## How to Run

1. Clone the repository (if applicable):
   ```bash
   git clone <repository-url>
   cd fashion_mnist_project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```

This will:
- Download the Fashion MNIST dataset (if not already present in ./data).
- Perform EDA and save a class distribution plot.
- Run experiments to find the best topology, optimizer, learning rate, and activation function.
- Train a final model and save it as final_best_model.pth.
- Generate visualizations (e.g., confusion_matrix.png, sample_predictions.png).

## Output
The script produces the following files:
- eda_class_distribution.png: Class distribution of the training set.
- learning_curves.png: Training loss and validation accuracy over epochs.
- activation_comparison.png: Bar plot comparing activation functions.
- confusion_matrix.png: Confusion matrix of the final model.
- sample_predictions.png: Sample predictions with true and predicted labels.
- best_model.pth: Best model checkpoint during training.
- final_best_model.pth: Final trained model.
