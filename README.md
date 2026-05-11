# Adult Income Classification with Neural Networks

A deep learning project that predicts whether an individual earns **more or less than $50K per year** using demographic and employment data from the UCI Adult (Census Income) dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

This project builds a **binary classification neural network** to predict adult income levels using PyTorch. It includes full data preprocessing, exploratory data analysis (EDA), and model training with evaluation.

**Target Variable:** `income` — whether an individual earns `>50K` or `<=50K` annually.

**Final Accuracy: ~83.8%**

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Samples:** ~32,500 (after deduplication)
- **Features:** 14 attributes (mix of numerical and categorical)

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Age of the individual |
| `workclass` | Categorical | Employment type |
| `fnlwgt` | Numerical | Final weight (census) |
| `education` | Categorical | Highest education level |
| `education.num` | Numerical | Education years |
| `marital.status` | Categorical | Marital status |
| `occupation` | Categorical | Job type |
| `relationship` | Categorical | Family role |
| `race` | Categorical | Race |
| `sex` | Categorical | Gender |
| `capital.gain` | Numerical | Capital gain |
| `capital.loss` | Numerical | Capital loss |
| `hours.per.week` | Numerical | Weekly work hours |
| `native.country` | Categorical | Country of origin |

---

## Project Structure

```
adult-income-classification/
|
|-- adult_income_classification.ipynb   # Main Jupyter notebook
|-- adult.zip                            # Raw dataset
|-- README.md                            # Project documentation
|-- requirements.txt                     # Python dependencies
```

---

## Features

- **Data Cleaning** — Duplicate removal, null value checks
- **Exploratory Data Analysis (EDA):**
  - Income distribution bar chart
  - Income by age group
  - Income by relationship status
  - Income by gender
  - Feature correlation heatmap
- **Data Preprocessing:**
  - Label encoding for target variable
  - One-hot encoding for categorical features
  - Standard scaling for numerical features
  - Scikit-learn `Pipeline` and `ColumnTransformer`
- **Neural Network Training** with PyTorch (mini-batch gradient descent)
- **Training Loss Visualization**

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/your-username/adult-income-classification.git
cd adult-income-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

1. **Download the dataset** from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult) and place `adult.zip` in the project root.

2. **Launch Jupyter Notebook:**

```bash
jupyter notebook adult_income_classification.ipynb
```

3. **Run all cells** in order — the notebook handles everything from data loading to model evaluation.

---

## Model Architecture

A fully connected feedforward neural network built with **PyTorch**:

```
Input Layer  ->  [input_size features]
     |
Hidden Layer 1  ->  64 neurons  +  ReLU activation
     |
Hidden Layer 2  ->  32 neurons  +  ReLU activation
     |
Output Layer  ->  1 neuron  +  Sigmoid activation
```

### Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 100 |
| Batch Size | 64 |

---

## Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **~83.8%** |

The model converges steadily over 100 epochs, showing consistent loss reduction throughout training.

---

## Visualizations

The notebook includes the following charts:

1. **Income Distribution** — Proportion of individuals earning <=50K vs >50K
2. **Income by Age Group** — Count plot across age brackets (19-30, 31-40, etc.)
3. **Income by Relationship** — Income breakdown across relationship categories
4. **Income by Gender** — Gender-based income comparison
5. **Correlation Heatmap** — Correlation matrix of all numerical features
6. **Training Loss Curve** — Loss per epoch over the full training run

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **PyTorch** | Neural network framework |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Preprocessing, pipelines, train/test split |
| **Matplotlib** | Plotting and visualization |
| **Seaborn** | Statistical visualizations |
| **Jupyter Notebook** | Interactive development environment |

---

## Acknowledgements

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
- Original data extracted from the 1994 US Census by Barry Becker
