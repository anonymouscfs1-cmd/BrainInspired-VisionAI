# BrainInspired-VisionAI
### Visual Pattern Recognition Inspired by the Human Brain

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

---

## Project Context

BrainInspired-VisionAI is a personal portfolio project exploring how artificial neural networks can process and interpret visual information in a way that is inspired by how the human brain works. The human brain does not process an entire visual scene at once — it selectively focuses on the most relevant parts of what it sees. This project investigates whether AI models can replicate that behaviour through attention mechanisms.

The project combines Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to classify images and generate interpretable visualisations that show which parts of an image influenced the model's prediction. The goal is not only to build a working classification system, but also to understand and explain what is happening inside the model — a field known as Explainable AI (XAI).

The project uses benchmark datasets such as CIFAR-10 and MNIST, and is structured in a way that makes it straightforward to extend to new datasets, models, or visualisation techniques in the future.

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [My Role and Contributions](#my-role-and-contributions)
- [Pipeline Overview](#pipeline-overview)
- [Development Steps](#development-steps)
- [Key Features](#key-features)
- [Technologies](#technologies)
- [Results and Findings](#results-and-findings)
- [Visualisations](#visualisations)
- [Challenges](#challenges)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

---

## Problem Definition

Standard image classification models — even when they achieve high accuracy — function as black boxes. They produce a prediction, but they do not explain why. For applications where understanding the model's reasoning matters, this is a significant limitation. A model could be achieving high accuracy for the wrong reasons: focusing on background noise, lighting artefacts, or irrelevant features rather than the actual object of interest.

This project addresses that problem. By applying Explainable AI techniques such as Grad-CAM and Integrated Gradients on top of trained CNN and ViT models, the system generates attention maps and heatmaps that show which pixels or regions drove the model's decision. These visualisations make it possible to inspect the model's behaviour, identify weaknesses, and understand what the network has actually learned.

The central question guiding this project is: *Can we build an image recognition system that not only classifies correctly, but also shows its reasoning in a way that is interpretable to a human?*

---

## My Role and Contributions

This is a fully solo project. All work — including dataset selection, preprocessing, model design, training, XAI integration, evaluation, and documentation — is my own independent contribution.

Specific responsibilities:

- Selecting and preparing datasets (CIFAR-10, MNIST, and optionally custom image sets)
- Designing and training a baseline CNN architecture
- Implementing a Vision Transformer (ViT) model and comparing it against the CNN baseline
- Applying Grad-CAM and Integrated Gradients to generate and visualise attention maps
- Evaluating model performance using accuracy, loss curves, and confusion matrices
- Analysing cases where model attention diverged from human intuition
- Writing structured, reproducible code and documentation

---

## Pipeline Overview

The core of this project is a visual recognition pipeline that takes an input image, classifies it, and then generates an interpretable explanation of how that classification was reached.

```
Input Image / Video
        |
Preprocessing
   Normalization, Resize, Augmentation (rotations, flips, crops)
        |
Model Inference
   CNN  -  5-layer: Conv > ReLU > MaxPool > Fully Connected
   ViT  -  Patch Embedding + Multi-Head Self-Attention
        |
Prediction Output
   Predicted class label + confidence score
        |
Explainability Layer
   Grad-CAM            - pixel-level influence map per class
   Integrated Gradients - input attribution / feature importance
   ViT Attention Maps   - self-attention weights per image patch
        |
Visualisation Output
   Heatmap overlay on original image
   Side-by-side CNN vs ViT comparison
   Saved as images or GIF for documentation
```

---

## Development Steps

### Step 1 — Dataset Selection and Preprocessing

The project uses CIFAR-10 and MNIST as benchmark datasets. CIFAR-10 was chosen because it contains 10 visually distinct object classes with meaningful spatial features, making it a useful testbed for attention visualisation. MNIST was included as a simpler baseline to verify that the pipeline runs correctly before moving to more complex data.

Preprocessing steps include:

- Normalisation of pixel values to the [0, 1] range
- Resizing to uniform input dimensions
- Data augmentation: random horizontal flips, rotations, and random crops to improve generalisation and reduce overfitting

A sample visualisation step was added to inspect random batches with class labels before training begins, to catch annotation or loading errors early.

### Step 2 — Baseline CNN Implementation

A 5-layer CNN was built from scratch using PyTorch. The architecture follows a standard pattern: Convolutional layer, ReLU activation, MaxPooling, repeated across two blocks, followed by fully connected layers for classification.

This baseline serves two purposes. First, it establishes a performance benchmark before introducing more complex architectures. Second, it provides a model that works well with Grad-CAM, since the convolutional feature maps are directly compatible with that technique.

Training runs were logged with accuracy and loss curves per epoch for both the training and validation sets. Model checkpoints were saved at the best validation accuracy.

### Step 3 — Vision Transformer (ViT) Implementation

A Vision Transformer was implemented as the second architecture. The ViT divides the input image into fixed-size patches, embeds each patch as a vector, and processes the sequence of patch embeddings through multi-head self-attention layers.

This architecture is fundamentally different from CNNs in how it captures spatial relationships. While CNNs use local receptive fields and build up spatial hierarchies layer by layer, ViTs can relate distant parts of an image directly through attention. This difference becomes visible when comparing the attention maps of both models on the same image.

The ViT was trained under the same conditions as the CNN baseline to allow a fair comparison. Training on small datasets such as CIFAR-10 is known to be challenging for ViTs, and this project documented that difficulty as part of the findings.

### Step 4 — Explainable AI Integration

Two XAI techniques were applied after training:

**Grad-CAM (Gradient-weighted Class Activation Mapping)** works by computing the gradient of the class score with respect to the feature maps in the final convolutional layer. These gradients are used to weight the feature maps, and the result is a coarse heatmap showing which regions of the image were most influential for a given prediction. Grad-CAM is compatible with any CNN architecture and requires no changes to the model during training.

**Integrated Gradients** is a model-agnostic attribution method that works by integrating the gradients of the model output with respect to the input along a path from a baseline (typically a black image) to the actual input. It assigns an importance score to each input feature (pixel) and is more precise than Grad-CAM but also more computationally expensive.

Both techniques were applied on held-out test images across all 10 CIFAR-10 classes to produce interpretable heatmaps.

### Step 5 — Attention Map Analysis

After generating attention maps for a representative set of test images, the visualisations were examined to understand what the model had learned. Several categories of behaviour were identified:

- Cases where the model correctly focused on the primary object
- Cases where the model focused on background regions or textures rather than the object itself
- Cases where attention was diffuse and spread across the entire image without clear focus

The second and third categories are particularly informative. They indicate either that the model has not fully separated the object from its context, or that the model is using spurious correlations in the training data to make its predictions. These findings are documented as part of the results section.

### Step 6 — Pipeline Packaging and Reproducibility

The codebase was organised into separate modules with clear responsibilities:

- `dataset_loader.py` — handles dataset downloading, preprocessing, and batch creation
- `models/cnn.py` — CNN architecture definition and forward pass
- `models/vit.py` — ViT architecture definition and forward pass
- `train.py` — training loop, validation, checkpoint saving, and metric logging
- `visualise.py` — Grad-CAM, Integrated Gradients, and attention map generation

An interactive Google Colab notebook was added to allow the full pipeline to be run without any local setup. The notebook walks through each step from data loading to attention visualisation.

---

## Key Features

- Dual architecture implementation: CNN and Vision Transformer trained and compared under identical conditions
- Grad-CAM heatmaps generated for any input image, showing pixel-level influence per predicted class
- Integrated Gradients providing quantitative feature attribution scores
- ViT self-attention visualisation showing which image patches the model attended to
- Side-by-side comparison of CNN vs ViT attention on the same images
- Evaluation metrics: accuracy curves, confusion matrix, per-class performance breakdown
- Works on both still images and video frames
- Fully reproducible in Google Colab without a local GPU required
- Modular codebase designed to make it straightforward to swap in new models or datasets

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, torchvision |
| Computer Vision | OpenCV |
| Visualisation | Matplotlib, Seaborn |
| XAI | Grad-CAM, Integrated Gradients |
| Datasets | CIFAR-10, MNIST |
| Development | Google Colab, GitHub |

---

## Results and Findings

**Classification performance**

Both models achieved reasonable classification accuracy on the CIFAR-10 test set. The CNN trained faster and was more stable across augmentation settings. The ViT required more epochs to converge and was more sensitive to learning rate and batch size choices. On MNIST, both models achieved high accuracy quickly due to the relative simplicity of the task.

**Attention map quality**

Grad-CAM heatmaps were broadly consistent with human intuition for classes with distinctive shapes, such as aeroplanes and automobiles. For classes that are visually similar or share common backgrounds — such as cats and dogs — the attention maps were less stable and occasionally highlighted background regions rather than the animal itself.

Integrated Gradients produced more localised attribution maps than Grad-CAM, but were noisier and harder to interpret visually. They were most useful for quantifying feature importance rather than for producing clean visualisations.

ViT attention maps differed meaningfully from CNN Grad-CAM maps. The ViT spread attention more broadly across the image, which in some cases captured contextual information that the CNN missed, but in other cases resulted in less focused attribution.

**Attention instability as a diagnostic tool**

One of the more useful findings from this project is that unstable or counterintuitive attention maps serve as a diagnostic signal. When the model's attention diverges significantly from what a human would consider the relevant region, it indicates that the model may be relying on spurious correlations — or may not have generalised beyond the training distribution. This makes attention visualisation a practical tool for identifying model weaknesses, not just for producing illustrative outputs.

---

## Visualisations

### Step 1: Raw Input Image
Original image from the CIFAR-10 test set before preprocessing.

*(Add screenshot here)*

---

### Step 2: Preprocessed Batch
Normalised and augmented batch visualised with class labels, used to verify the data pipeline before training.

*(Add screenshot here)*

---

### Step 3: Training Curves
Accuracy and loss over epochs for both training and validation sets, for CNN and ViT side by side.

*(Add screenshot here)*

---

### Step 4: Grad-CAM Heatmap
Grad-CAM overlay on a test image showing which regions drove the prediction. Warmer colours indicate higher influence on the model's output.

*(Add screenshot here)*

---

### Step 5: CNN vs ViT Attention Comparison
Side-by-side comparison of Grad-CAM (CNN) and self-attention map (ViT) on the same input image, showing how the two architectures differ in their spatial focus.

*(Add screenshot here)*

---

### Step 6: Failure Case
An example where model attention focused on background rather than the target object. Cases like this are used to document where the model's learned features do not align with human intuition.

*(Add screenshot here)*

---

*A short demo GIF showing attention map generation across multiple test images will be added upon project completion.*

---

## Challenges

**Attention instability across similar classes**

Grad-CAM maps were inconsistent for visually similar classes in CIFAR-10, particularly cats and dogs. The model's decision boundary in these cases appears to involve background context and texture rather than the object itself, which produces unreliable attention maps. This is a known limitation of class activation mapping on datasets without precise segmentation labels.

**Interpreting ViT attention**

Vision Transformer attention maps are more complex to interpret than CNN Grad-CAM maps. Each attention head learns a different focus pattern, and aggregating across heads requires design choices (mean vs. max pooling over heads, for example) that directly affect the resulting visualisation. There is no single agreed standard for ViT attention visualisation, which makes comparisons across architectures less straightforward.

**Training ViTs on small datasets**

Vision Transformers are data-hungry by design. Training a ViT from scratch on CIFAR-10 without pre-training results in slower convergence and lower accuracy compared to CNNs. Data augmentation was applied aggressively and learning rate scheduling was used to address this, but the gap with the CNN baseline remained visible.

**Computational cost**

Running multiple model training runs alongside Grad-CAM and Integrated Gradients generation was computationally demanding on Colab's free tier. Long training runs required careful management of session time and checkpoint saving.

**High accuracy does not guarantee interpretable attention**

Even when classification accuracy was high, some attention maps pointed to irrelevant image regions. This confirmed that accuracy alone is not a sufficient quality signal for a model that is intended to be interpretable or explainable.

---

## How to Run

### Option 1: Google Colab

Open the notebook using the badge at the top of this README. The notebook runs end-to-end without any local setup. A free Colab GPU is sufficient for all CIFAR-10 and MNIST experiments.

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BrainInspired-VisionAI.git
cd BrainInspired-VisionAI

# Install dependencies
pip install -r requirements.txt

# Train a model
python src/train.py --dataset cifar10 --model cnn --epochs 30

# Generate attention maps on a test image
python src/visualise.py --input data/sample.jpg --model checkpoints/best_model.pt --method gradcam
```

### Repository Structure

```
BrainInspired-VisionAI/
├── notebooks/
│   └── demo.ipynb              # Interactive Colab demo
├── src/
│   ├── dataset_loader.py       # Dataset loading and preprocessing
│   ├── models/
│   │   ├── cnn.py              # CNN architecture
│   │   └── vit.py              # Vision Transformer architecture
│   ├── train.py                # Training loop and checkpoint saving
│   └── visualise.py            # Grad-CAM, Integrated Gradients, ViT attention
├── data/                       # Sample images
├── results/                    # Saved attention maps and metrics
├── requirements.txt
└── README.md
```

---

## Future Work

- Extend the project to larger and more challenging datasets such as CIFAR-100 or Tiny ImageNet
- Explore pre-trained ViT weights (e.g. from ImageNet) to improve ViT performance on small datasets
- Add real-time video input: process frames sequentially and display an attention map per frame as output
- Investigate capsule networks as a more biologically plausible alternative to standard CNNs
- Integrate automatic evaluation of attention quality using segmentation masks as ground truth, to objectively measure how well the model's focus aligns with the actual object
- Explore multimodal extensions: combining image attention with natural language descriptions of what the model focuses on
- Potential integration with drone footage for applied use cases such as crop monitoring — building directly on techniques explored in previous computer vision projects

---

## License

This project is licensed under the [MIT License](LICENSE).
