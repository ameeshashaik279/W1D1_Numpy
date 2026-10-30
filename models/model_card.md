# Model Card – Best Model

## 1. Model Name
Logistic Regression

## 2. Project
MLflow Experiment Tracking Platform

## 3. Training Data
- Dataset: Iris Dataset
- Total records: 150
- Training records: 120
- Testing records: 30
- Features: 4
- Target: 3 flower classes

## 4. Evaluation Metric
- Metric: Accuracy
- Accuracy Score: 93.33%

## 5. Model Selection
Logistic Regression was selected because it achieved the highest accuracy of 93.33%, tied with Decision Tree.

## 6. Known Limitations
- The model was evaluated using a small Iris dataset.
- Performance on other datasets is not guaranteed.
- The model may not perform well on unseen data with different distributions.
- Further testing is required before production deployment.

## 7. Model File
Saved as: `models/best_model.pkl`

## 8. Conclusion
Logistic Regression achieved 93.33% accuracy and was selected as the best model for this project.
## 9. Hyperparameter Tuning – W4D2

**Technique:** RandomizedSearchCV

**Cross-validation:** 5-fold

**Number of iterations:** 20

### Best Parameters

- C: 100
- Solver: lbfgs
- Max iterations: 500

### Performance

- Previous CV accuracy: 95.83%
- Tuned CV accuracy: 96.67%
- Improvement: 0.84 percentage points

### Tuned Model

Saved as: `models/tuned_model.pkl`

The tuned Logistic Regression model achieved improved cross-validation accuracy compared with the baseline cross-validation result.
