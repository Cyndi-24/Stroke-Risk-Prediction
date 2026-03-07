# Stroke Risk Prediction: A Machine Learning Model
## Presented by Team Oracle

# Project Overview

A stroke is a medical emergency that occurs when blood flow to a part of the brain is interrupted or a weakened blood vessel in the brain bursts, causing brain cells to die within minutes. Stroke has been found to be one of the leading causes of death.It requires immediate emergency care to prevent permanent brain damage.Common warning signs of stroke range from loss of balance,vision changes, facial drooping ,arm and leg weakness and speech difficulty.

According to World Health Organisation ,millions of people suffer stroke every year many of which could have been avoided through early detection and proper risk monitoring.Major risk factors for stroke include high blood pressure (the leading cause), smoking, diabetes, high cholesterol, obesity, and atrial fibrillation. 

The purpose of this study is  to develope a machine learning model able to identity patients at high risk of stroke using lifestyle ,demographic and medical indicators.With this, high risk patients will be identified early for immediate intervetions

Multiple machine learning models were implemented ,compared and the best was selected. These models include; Logistic Regression model,Decision Trees , and Random Forest models.The model with the best performance was selected.

-----
# Problem Statement

Despite the availability of medical indicators that depict increased risk of stroke,predicting its occurence remains difficult due to complex relationship between between demographic,lifestyle and clinical factors.This challenge was addressed by performing an exploratory data analysis on the data set and developing a machine learning classification model to predict stroke risk thus supporting early intervention through data-driven insights.

----
# Objectives

* Data cleaning and wrangling
* Perform Exploratoty Data Analysis
* Identification of factors associated with stroke
* Feature Engineering
* Build a suitable classification model for analysis
* Evaluate model performance
* Recommendations and Insights for early intervention

----
# Data Sourcing
 The data set for this project was sourced from kaggle.com
 [https://www.kaggle.com/fedesoriano/stroke-prediction-dataset]

 ---
 # Dataset Overview
 The data set consist of 5,110 rows and 12 columns with each row representing a unique patient and the columns containing information on the demographics and clinical features of each patient such as ; age,gender,hypertension,average glucose level,BMI,smoking status, heart disease and work type

----
# Tools Used
* Python
* Jupyter Notebook
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Scikit-learn

---
# Machine Learning Models Used
* Logistic Regression: This is the most commonly used model for binary classification problems where the outcome has two possible results and in  this case the outcome variable is stroke or no stroke .This makes the logistic Regression a suitable starting model for predicting the probability of having a stroke and understanding how different health factors affect stroke risk.
  
* Decision Tree:This model mimics human decision making by splitting data into different branches based on feature value and how the contribute to stroke risk.It handles both numerical and categorical data and and captures non-linear relationship between variables.

* Random Forest:This builds multiple decision trees and combines their predictions to improve accuracy and reliability compared to other models.This model reduces overfitting  and is able to handle larger datasets and many features as well.It is most likely to identify influential risk factors.
