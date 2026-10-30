# Model Performance Report – W4D3

## 1. Model
Tuned Logistic Regression

## 2. Dataset
- Dataset: Iris
- Training samples: 120
- Testing samples: 30
- Number of features: 4
- Number of classes: 3

## 3. Evaluation Metrics

| Metric | Score | Interpretation |
|---|---:|---|
| Accuracy | 100% | All test predictions were correct |
| Precision | 100% | All predicted classes were correct |
| Recall | 100% | All actual classes were identified |
| F1-score | 100% | Precision and recall were both high |
| ROC-AUC | 100% | Excellent class discrimination on this test set |

## 4. Confusion Matrix
The confusion matrix was generated and saved as:

`models/confusion_matrix.png`

## 5. Optimisation Metric

Primary metric: Accuracy.

Reason: The Iris dataset has three classes, and the evaluation set is balanced. Accuracy provides a useful overall measure of correct predictions.

Precision, Recall, F1-score and ROC-AUC are also reported for a broader evaluation.

## 6. Conclusion

The tuned Logistic Regression model achieved 100% accuracy, precision, recall, F1-score and ROC-AUC on the 30-sample test set.

Further validation on additional unseen data is recommended before production deployment.
