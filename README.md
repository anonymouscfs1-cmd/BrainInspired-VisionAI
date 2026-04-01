BrainInspired-VisionAI
Project Context

BrainInspired-VisionAI is a solo research project exploring how artificial neural networks process and interpret visual information in a manner inspired by the human brain. By combining Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), the project investigates attention mechanisms, interpretability, and explainable AI (XAI) in image recognition tasks.

The project focuses on understanding AI decision-making, creating attention maps that show what parts of an image influence predictions, and bridging the gap between biological vision principles and modern AI architectures. This repository contains all my independent contributions and serves as a portfolio project for a potential Master’s in AI and PhD research in AI & neuroscience.

My Role & Contributions

Since this project is fully solo, I was responsible for:

Dataset Selection & Preprocessing: Loading, cleaning, augmenting, and splitting datasets (CIFAR-10, MNIST, and custom image datasets).
Model Design & Training: Implementing baseline CNNs and advanced ViT architectures for comparison.
Explainable AI Techniques: Applying Grad-CAM, Integrated Gradients, and attention visualization to interpret model predictions.
Evaluation & Benchmarking: Tracking accuracy, loss, and model attention stability.
Documentation & GitHub Workflow: Writing clear, structured code and README documentation suitable for professional portfolio presentation.
Table of Contents
Visual Recognition Pipeline
Key Features
Technologies
Development Steps
Results & Findings
Challenges
Visualizations
Future Work
Credits
License
Visual Recognition Pipeline

The core of this project is a brain-inspired visual recognition pipeline designed to classify images and generate interpretable attention maps.

Pipeline Overview
Input Image / Video
        ↓
Preprocessing (normalization, augmentation)
        ↓
CNN / Vision Transformer Model
        ↓
Prediction (class label)
        ↓
Attention Map / Grad-CAM / Integrated Gradients
        ↓
Visualization Output (heatmap overlay)
Development Steps

Step 1 – Dataset Selection & Preprocessing

Chose benchmark datasets: CIFAR-10, MNIST, plus optional custom image datasets.
Normalized pixel values and applied augmentation (rotations, flips, crops) to reduce overfitting.

Step 2 – Baseline CNN Implementation

Built a 5-layer CNN with convolution, ReLU, pooling, and fully connected layers.
Trained and validated model to establish baseline accuracy.

Step 3 – Vision Transformer (ViT) Implementation

Developed a ViT model with patch embedding and self-attention layers.
Compared performance against CNN baseline.

Step 4 – Explainable AI Integration

Applied Grad-CAM to visualize which pixels influenced the model’s predictions.
Tested Integrated Gradients to quantify feature importance.

Step 5 – Attention Map Analysis

Visualized attention maps for different classes to understand model focus.
Analyzed discrepancies where AI attention diverged from human intuition.

Step 6 – Pipeline Packaging & Reproducibility

Organized code into reusable modules: dataset loader, model trainer, attention visualizer.
Added notebooks demonstrating step-by-step workflow.
Key Features
Works on images and videos with attention visualization.
Implements both CNN and Vision Transformer architectures.
Generates interpretable attention maps for each prediction.
Modular pipeline for easy experimentation with new models.
Fully reproducible in Google Colab or local Python environment.
Technologies
Python
PyTorch (model training and evaluation)
OpenCV & Matplotlib (image processing and visualization)
Grad-CAM / Integrated Gradients (XAI techniques)
Google Colab (development and demo environment)
Results & Findings
Achieved robust classification on CIFAR-10 and MNIST datasets.
Attention maps highlighted relevant image regions, aligning with human intuition in most cases.
CNNs provided faster training with lower computational cost.
ViTs demonstrated better long-range attention capture but required more resources.
The project demonstrates that attention-based visualization can reveal model weaknesses, such as overfitting to irrelevant features.
Challenges
Attention maps are sometimes unstable or misleading, especially for similar classes.
Limited dataset sizes can reduce generalization, requiring extensive augmentation.
Interpreting ViT attention is more complex than CNN Grad-CAM maps.
High computational demand when testing multiple architectures on large datasets.
Visualizations

Step 1: Raw Image
Original images from CIFAR-10/MNIST datasets.

Step 2: Prediction Output
Model prediction label.

Step 3: Attention Heatmap
Grad-CAM or Integrated Gradients overlaid to show influential regions.

Step 4: Multi-image Visualization
Side-by-side comparison of CNN vs ViT attention maps.

Future Work
Extend to larger datasets (CIFAR-100, Tiny ImageNet).
Explore brain-inspired network architectures, like capsule networks.
Combine visual reasoning with text or multimodal inputs.
Integrate automatic evaluation metrics for attention correctness.
Explore collaboration with AI research tools (Claude AI, GPT, etc.) for automated insight generation.
Credits
All code, experiments, and documentation are my independent work.
Libraries and frameworks: PyTorch, OpenCV, Matplotlib.
Datasets: CIFAR-10, MNIST, and other open-source datasets.
License

MIT License
