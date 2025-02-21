Overview

This Jupyter Notebook applies a Logistic Regression model to classify loan applicants into two categories:

0 (Healthy Loan): Low-risk applicants likely to repay their loans.

1 (High-Risk Loan): High-risk applicants who may default on their loans.

Dataset

The dataset contains features related to loan applications, including financial indicators, borrower details, and credit history. The target variable is funding_status, where:

0 represents a healthy loan

1 represents a high-risk loan

Workflow

The notebook follows these key steps:

Load and Preprocess Data

Read the CSV file into a Pandas DataFrame.

Identify and handle missing values (if any).

Separate the data into features (X) and labels (y).

Split the Data

Use train_test_split to divide the data into training (80%) and testing (20%) sets.

Set random_state=1 for reproducibility.

Train the Logistic Regression Model

Instantiate a Logistic Regression model from sklearn.linear_model.

Fit the model using the training dataset (X_train, y_train).

Make Predictions

Use model.predict(X_test) to generate predictions on the test dataset.

Evaluate Model Performance

Compute a confusion matrix to analyze correct and incorrect predictions.

Print a classification report to assess precision, recall, and F1-score.

Model Performance

Classification Report Summary:

Healthy Loan (0): Precision = 1.00, Recall = 0.99, F1-score = 1.00

High-Risk Loan (1): Precision = 0.84, Recall = 0.94, F1-score = 0.89

Overall Accuracy: 99%

Confusion Matrix:

[[18655,   110]
 [   36,   583]]

18655 True Negatives (correctly classified healthy loans)

583 True Positives (correctly classified high-risk loans)

110 False Positives (healthy loans misclassified as high-risk)

36 False Negatives (high-risk loans misclassified as healthy)

Key Insights

The model excels in identifying healthy loans, achieving nearly perfect performance.

The model performs well on high-risk loans, but with slightly lower precision (84%), meaning some healthy loans are misclassified as high-risk.

High recall (94%) for high-risk loans ensures that most actual high-risk cases are detected.

Potential Improvements

Feature Engineering: Add or refine features that better distinguish high-risk borrowers.

Hyperparameter Tuning: Experiment with different regularization strengths for better balance between precision and recall.

Resampling Methods: Address class imbalance (if present) using techniques like SMOTE (Synthetic Minority Over-sampling Technique).

Dependencies

This notebook requires the following Python libraries:

pandas

numpy

sklearn

How to Use

Clone the repository or download the notebook.

Ensure all dependencies are installed (pip install pandas numpy scikit-learn).

Run all cells in the notebook to train and evaluate the model.

Author

This notebook was developed for analyzing credit risk classification using machine learning techniques.

End of README# credit-risk-classification
