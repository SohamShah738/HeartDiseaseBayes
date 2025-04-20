# Bayesian Network for Heart Disease Prediction

This project uses a **Bayesian Network** (built with `pgmpy`) to model and predict the risk of heart disease based on patient data.

## Features

- Manual and learned Bayesian network structure comparison
- Discrete data preprocessing using domain-based binning
- Model training with Laplace smoothing (Bayesian Estimation with BDeu)
- Inference engine to predict heart disease given symptoms
- Full evaluation with accuracy, precision, recall, F1 score, and confusion matrix

## How it Works

1. Data is loaded from the UCI Heart Disease dataset (`data/heart.csv`)
2. Features like age, cholesterol, and blood pressure are discretized
3. Two models are trained:
   - A manually defined Bayesian network
   - A learned structure via Hill Climb Search
4. Both models are evaluated on a test set
