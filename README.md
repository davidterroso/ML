# 📘 Machine Learning – Math Foundations

A comprehensive learning resource focused on explaining and implementing the **mathematical foundations** behind key machine learning algorithms. Each notebook walks through the theory step-by-step, along with code implementations and visualizations for deeper understanding.

## 📂 Project Structure

```text
ML/
├── README.md
├── pyproject.toml          # Project dependencies (managed with uv)
├── .python-version         # Python 3.12.3
├── regression.ipynb        # Linear regression & cost functions
├── classification/         # Classification algorithms
│   ├── logistic_regression.ipynb
│   ├── decision_tree.ipynb
│   ├── naive_bayes.ipynb
│   ├── neighbours.ipynb    # k-Nearest Neighbors & Radius-based methods
│   ├── random_trees.ipynb
│   ├── svm.ipynb           # Support Vector Machines
│   └── classification.md
└── metrics/                # Evaluation & distance metrics
    └── distances.ipynb
```

## 📚 Notebooks

### Regression

- **`regression.ipynb`** – Comprehensive guide to regression:
  - Linear Regression (Normal Equation and Gradient Descent)
  - Cost Functions and Loss Minimization
  - Model Evaluation Metrics

### Classification

Classification algorithms for supervised learning with labeled data:

- **`logistic_regression.ipynb`** – Logistic Regression
  - Binary classification fundamentals
  - Decision boundaries
  - Gradient descent in classification

- **`neighbours.ipynb`** – Distance-based Methods
  - k-Nearest Neighbors (kNN)
  - Radius-based Nearest Neighbors
  - KD-tree optimizations
  - Euclidean and other distance metrics

- **`naive_bayes.ipynb`** – Naive Bayes Classification
  - Probabilistic approach to classification
  - Gaussian Naive Bayes

- **`decision_tree.ipynb`** – Decision Trees
  - Tree-based decision making
  - Information gain and entropy

- **`random_trees.ipynb`** – Random Forests
  - Ensemble methods
  - Bootstrap aggregation

- **`svm.ipynb`** – Support Vector Machines
  - Kernel methods
  - Maximum margin classification

### Metrics & Evaluation

- **`distances.ipynb`** – Distance Metrics
  - Euclidean, Manhattan, Cosine distances
  - Distance calculations for ML algorithms
  - Visualization and comparison

## 🎯 Purpose

The goal is to **demystify the math** behind machine learning by combining:

- Mathematical formulas and derivations
- Conceptual explanations
- Hands-on Python implementations
- Visual demonstrations

Ideal for students, practitioners, or anyone wanting to strengthen their theoretical foundation in ML.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (project uses Python 3.12.3)
- `uv` package manager (fast, modern Python package management)

### Setup

1. **Install uv** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and navigate to the project**:

   ```bash
   cd ML
   ```

3. **Install dependencies**:

   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:

   ```bash
   source .venv/bin/activate
   ```

5. **Launch Jupyter Lab**:

   ```bash
   jupyter lab
   ```

### Development Dependencies

To install additional development tools (pytest, linting, type checking):

```bash
uv sync --extra dev
```

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## 📌 Future Topics

- Principal Component Analysis (PCA)
- Clustering algorithms (K-means, Hierarchical)
- Dimensionality reduction
- Neural Networks (from scratch)
- Ensemble methods (Boosting, Bagging)

## 📚 License

This project is open-source and available under the MIT License.

---

Explore, learn, and strengthen your understanding of ML fundamentals!
