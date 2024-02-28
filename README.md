# CreditScoring-Analysis
This project aims to develop a machine learning model for banks to evaluate the creditworthiness of potential subprime mortgage borrowers, enabling data-driven lending decisions, market expansion, and profit maximization.

Objectives:
Develop an in-house risk model for subprime mortgage lending decisions, 
Maximize profit while maintaining market competitiveness.

Data Description

Dataset: Historical loan applications (approximately 3000 observations, 30 variables)
Target Variable: Binary variable (0 = Good loan, 1 = Bad loan)
Additional Information:
Profit per good loan: $100
Loss per bad loan: $500

Methodology

Data Preprocessing:
Clean and prepare the data for analysis (handling missing values, outliers, etc.)

Feature engineering (create additional features for improved model performance)

Feature selection (identify the most relevant features)

Model Training:
Utilized logistic regression to train a classification model based on 80% of the labeled data.

Optimize hyperparameters (parameters within the model itself) for optimal performance.

Model Evaluation:
Assess model performance using metrics like accuracy, precision, recall, F1-score, and AUC-ROC curve.
Interpret model coefficients to understand the impact of each feature on loan repayment probability.

Decile Methodology for Business Strategy:

Divide the remaining 20% of data (sorted by descending predicted good loan probability) into 10 equal deciles (groups).
Calculate the cumulative good loan percentage (sensitivity) and cumulative bad loan avoided percentage (specificity) for each decile.
Plot these values on a ROC curve to visualize model performance.
Calculate the profit/loss for each decile based on predicted good/bad loan probabilities and associated financial penalties/rewards.
Analyze the trade-off between risk and reward for different loan acceptance thresholds within each decile.
