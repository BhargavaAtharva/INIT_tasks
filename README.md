# Research Task 2 – Supervised & Unsupervised Learning

## Dataset Source
The dataset used in this task is the **Life Expectancy Dataset** published by the World Health Organization (WHO) and made available on Kaggle.  
Dataset Source: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/data

---

## Problem Statement
The objective of this task is twofold:

1. **Supervised Learning**  
   To predict a country’s life expectancy using multiple health and socio-economic indicators by implementing **Linear Regression from scratch**, and comparing its performance with a scikit-learn implementation.

2. **Unsupervised Learning**  
   To discover underlying structure in the dataset by grouping countries based on selected indicators using **K-Means clustering from scratch**, without using life expectancy as a label.

---

## Execution Steps

1. Load the Life Expectancy dataset into a pandas DataFrame `df`.

2. Inspect missing values using:
   df.isnull().sum()
   and handle missing values using median imputation for numerical columns.

3. Preprocess the data:
   - Drop non-informative identifier columns such as `Country`.
   - Encode the categorical column `Status` using binary encoding.
   - Separate numerical and categorical features.

4. Feature scaling:
   - Standardize numerical features using StandardScaler.
   - Apply scaling only after train-test split to avoid data leakage.

5. Custom train-test split:
   - Shuffle the dataset using a fixed random seed.
   - Split the data into 80% training and 20% testing sets.

6. Separate features and target:
   - X = all columns except `Life expectancy`
   - y = `Life expectancy`

7. Train custom Linear Regression model (from scratch):
   - Implement hypothesis function.
   - Define loss function (Mean Squared Error).
   - Implement gradient descent for weight updates.
   - Train the model using fixed learning rate and iterations.

8. Evaluate Linear Regression:
   - Predict life expectancy on training and testing sets.
   - Compute Mean Absolute Error (MAE) for evaluation.
   - Compare performance with scikit-learn’s LinearRegression model.

9. Train custom K-Means clustering model (unsupervised):
   - Select relevant health and socio-economic features.
   - Initialize k = 3 centroids randomly from the training data.
   - Assign each data point to the nearest centroid using Euclidean distance.
   - Update centroids as the mean of assigned cluster points.
   - Repeat assignment and update steps until convergence or fixed iterations.

10. Evaluate K-Means clustering:
    - Predict cluster indices for all data points.
    - Visualize clusters using PCA-based 2D scatter plot.
    - Perform post-hoc analysis by comparing cluster groups with average life expectancy.

11. Interpret results:
    - Analyze structure revealed by clustering.
    - Discuss alignment with life expectancy trends.
    - Explain how unsupervised insights help understand supervised model limitations.

---

## Short Summary
In this task, Linear Regression was implemented from scratch to predict life expectancy and evaluated using Mean Absolute Error (MAE), with results closely matching the scikit-learn baseline.  
K-Means clustering was also implemented from scratch with three clusters, revealing meaningful groupings of countries based on health and socio-economic characteristics. PCA was used only for visualization to interpret cluster structure.  
Overall, the task demonstrates both supervised and unsupervised learning workflows, along with insights into model limitations and data heterogeneity.

##Video 
https://drive.google.com/file/d/1X2kqIMjJK6NufZODbrOu_WeunONalQ2c/view?usp=sharing