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
# Exploratory Data Analysis(EDA) 
## Data Manipulation Using Pandas and Numpy Libraries
* Library Importation

 
  
* Dataset Importation



* Data Cleaning:
  Null Values were identified in the BMI column and was filled with the median value


## Data Visualization and Distributions Using Matplotlib and Seaborn Libraries
*  Work Type with Residence among People with Stroke



Insight: This countplot shows that people across all work categories in the private sector had the highest number of stroke cases especially in the urban area. This could be due to higher stress roles in this regions while in the pediatric population there are much lower stroke cases.

* Residence type with Smoking and Stroke


Insight: This barplot shows people who have been previously or currently  married have a higher probability of stroke across all smoking categories compared to those who are not married likely due to the fact that the married population are generally older than the unmarried population showing that age is big risk factor to stroke

* Gender Distribution


Insight: The distribution shows that the gender with the higher stroke population is the females.This could also arise due to more females participating in the study.

* Stroke vs Non-Stroke


Insight:This shows an imbalance between the population with stroke (4.31%) and those without stroke(95.69%) and with this SMOTE Analysis was introduced to the analysis to bring a balance between both

* Age vs BMI


Insight: The BMI range is narrower in children and increases as people attain adulthood.People between 20 and 80 years have BMI between 20 and 40

* Stroke vs age


Insight:Stroke is either 0(NO) or 1(YES).At the point where stroke is = 1,the dots are lighter for younger people but solid around 40-50yrs indicating that age is a major determinant of stroke.

Pair plot Correlation 


Insight:This shows that stroke cases are heavily concentrated among older individuals and those with higher glucose levels.Thus showing a weak positive correlation among the three values  with a value 0.132 but this value cannot be ignored because this is a medical dataset

Distribution of Medical Features 


Insight:The distribution of both hypertension and heart disease are heavily skewed towards 0 as a result of the class imbalance  and this was addressed during the modelling phase using SMOTE.

# Machine Learning Overview 

In this project, several machine learning algorithms were implemented to predict the likelihood of a patient experiencing a stroke based on demographic and health-related features. These models were selected to compare performance across linear, probabilistic, and ensemble-based approaches and the best performing model was selected.

## Models Used

* Logistic Regression: This is the most commonly used model for binary classification problems where the outcome has two possible results and in  this case the outcome variable is stroke or no stroke .This makes the logistic Regression a suitable starting model for predicting the probability of having a stroke and understanding how different health factors affect stroke risk.
  

* Random Forest:This builds multiple decision trees and combines their predictions to improve accuracy and reliability compared to other models.This model reduces overfitting  and is able to handle larger datasets and many features as well.

** Decision Tree:This model mimics human decision making by splitting data into different branches based on feature value and how the contribute to stroke risk.It handles both numerical and categorical data and and captures non-linear relationship between variables.
 
---
