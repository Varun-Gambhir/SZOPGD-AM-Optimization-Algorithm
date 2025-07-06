# Fashion MNIST Optimization with SZOPGD-AM and Adam

This project implements and compares the performance of a custom Stochastic Zeroth-Order Proximal Gradient Descent with Adaptive Momentum (SZOPGD-AM) optimizer against the standard Adam optimizer on the Fashion MNIST dataset using a ResNet18 model. The project includes training, evaluation, and visualization of training loss, training accuracy, and test accuracy for both optimizers.

## Project Structure

```
fashion_mnist_optimization/
├── src/
│   ├── optimizers/
│   │   └── szopgdam.py        # SZOPGD-AM optimizer implementation
│   ├── models/
│   │   └── resnet18.py        # ResNet18 model architecture
│   ├── utils/
│   │   └── training.py        # Training and evaluation functions
│   └── main.py                # Main script to run training and comparison
├── saved_models/
│   ├── resnet_fashion_mnist_szopgdam.pth  # Saved SZOPGD-AM model weights
│   └── resnet_fashion_mnist_adam.pth      # Saved Adam model weights
├── requirements.txt           # Project dependencies
├── README.md                 # Project documentation
└── optimizer_comparison.png   # Plot comparing optimizer performance
```

## Prerequisites

- Python 3.8+
- PyTorch 2.0.1
- Torchvision 0.15.2
- Matplotlib 3.7.5
- A CUDA-enabled GPU (optional, for faster training)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fashion_mnist_optimization.git
   cd fashion_mnist_optimization
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create the saved_models directory**:
   ```bash
   mkdir saved_models
   ```

## Usage

To train the models and compare SZOPGD-AM and Adam optimizers on Fashion MNIST:

```bash
python src/main.py
```

This will:
- Download the Fashion MNIST dataset
- Train two ResNet18 models: one with SZOPGD-AM and one with Adam
- Save the trained model weights to `saved_models/`
- Generate a plot (`optimizer_comparison.png`) comparing training loss, training accuracy, and test accuracy
- Print final test accuracies for both optimizers

## Results

The training process runs for 20 epochs and produces:
- A plot (`optimizer_comparison.png`) with three subplots:
  - Training Loss vs. Epoch
  - Training Accuracy vs. Epoch
  - Test Accuracy vs. Epoch
- Final test accuracies for both SZOPGD-AM and Adam optimizers
- Saved model weights in `saved_models/`

Sample output from the notebook:
- SZOPGD-AM Final Test Accuracy: ~90.84%
- Adam Final Test Accuracy: ~92.32%

## Notes

- The SZOPGD-AM optimizer is implemented with an adaptive learning rate (`eta0 / sqrt(step)`) and momentum, but uses first-order gradients in this implementation since PyTorch computes them automatically.
- The ResNet18 model is adapted for Fashion MNIST (grayscale images, 1 channel) with 10 output classes.
- Training is performed on GPU if available; otherwise, it falls back to CPU.
- The `saved_models/` directory will store the best model weights based on test accuracy.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

## License

This project is licensed under the MIT License.