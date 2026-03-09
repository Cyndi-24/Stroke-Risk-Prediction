# Stroke Risk Prediction: A Machine Learning Model
## Presented by Team Oracle

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/stroke_prediction_image.png)

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
  
![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/libraries_importation.png)
 
  
* Dataset Importation

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/data_importation.png)

* Data Cleaning

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/data_cleaning.png)

## Data Visualization and Distributions Using Matplotlib and Seaborn Libraries
*  Work Type with Residence among People with Stroke
  
![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/Work_type%20residence_with_stroke%20.png)

Insight: This countplot shows that people across all work categories in the private sector had the highest number of stroke cases especially in the urban area. This could be due to higher stress roles in this regions while in the pediatric population there are much lower stroke cases.

* Residence type with Smoking and Stroke

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/married_status_with_smoking.png)

Insight: This barplot shows people who have been previously or currently  married have a higher probability of stroke across all smoking categories compared to those who are not married likely due to the fact that the married population are generally older than the unmarried population showing that age is big risk factor to stroke

* Gender Distribution

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/distribution_by_gender.png)

Insight: The distribution shows that the gender with the higher stroke population is the females.This could also arise due to more females participating in the study.

* Stroke vs Non-Stroke

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/strokevsnonstrokke.png)

Insight:This shows an imbalance between the population with stroke (4.31%) and those without stroke(95.69%) and with this SMOTE Analysis was introduced to the analysis to bring a balance between both classes

* Age vs BMI

![image alt](https://github.com/Cyndi-24/Stroke-Risk-Prediction/blob/main/Stroke_prediction/Stroke_prediction_images/age_vs_bmi.png)

Insight: The BMI range is narrower in children and increases as people attain adulthood.People between 20 and 80 years have BMI between 20 and 40

* Stroke vs age

![image alt](

Insight:Stroke is either 0(NO) or 1(YES).At the point where stroke is = 1,the dots are lighter for younger people but solid around 40-50yrs indicating that age is a major determinant of stroke.

* Pair plot Correlation 


Insight:This shows that stroke cases are heavily concentrated among older individuals and those with higher glucose levels.Thus showing a weak positive correlation among the three values  with a value 0.132 but this value cannot be ignored because this is a medical dataset

* Distribution of Medical Features 


Insight:The distribution of both hypertension and heart disease are heavily skewed towards 0 as a result of the class imbalance  and this was addressed during the modelling phase using SMOTE.

# Machine Learning 

In this project, several machine learning algorithms were implemented to predict the likelihood of a patient experiencing a stroke based on demographic and health-related features. These models were selected to compare performance across linear, probabilistic, and ensemble-based approaches and the best performing model was selected.

## Models Used

* Logistic Regression: This is the most commonly used model for binary classification problems where the outcome has two possible results and in  this case the outcome variable is stroke or no stroke .This makes the logistic Regression a suitable starting model for predicting the probability of having a stroke and understanding how different health factors affect stroke risk.
* Random Forest:This builds multiple decision trees and combines their predictions to improve accuracy and reliability compared to other models.This model reduces overfitting  and is able to handle larger datasets and many features as well.
  
* Decision Tree: This model mimics human decision making by splitting data into different branches based on feature value and how the contribute to stroke risk.It handles both numerical and categorical data and and captures non-linear relationship between variables.

In this project, *the Decision Tree model* performed better than the other models tested, making it the most effective algorithm for predicting stroke risk. 

---
## Data Preparation For Modelling



The following steps were taken to properly preprocess the dataset and to ensure it was suitable enough  for model training:

a)The missing values under the BMI column were filled with the median value of thw BMI

b)The column and stroke ID column were dropped 

c)The categorical variables such as; gender,smoking status,ever-married,work-type,residence-type,were encoded

d)The dataset contains far fewer stroke cases(4.13%) compared to non-stroke cases(95.69%), which creates a class imbalance problem. If not handled, the model may become biased toward predicting the majority class.To address this, SMOTE (Synthetic Minority Oversampling Technique) was used

e)The dataset was split into training and test using the 80/20 ratio to train the machine learning model and to evaluate the model's performance on unseen data respectively.

---

## Model Evaluation - Decision Tree Classifier



After training the Decision Tree model, its performance was evaluated using a classification report, which includes precision, recall, F1-score, and accuracy. These metrics help assess how well the model predicts stroke and non-stroke cases.

* Accuracy:
The model achieved an overall accuracy of 91%, meaning that 91% of all predictions made by the model were correct showing strong overall performance in classifying patients into stroke and non-stroke categories.

* Precision: Precision measures how many of the predicted cases were actually correct.
 Class 0 (No Stroke): Precision = 0.92
 This means that 92% of the instances predicted as non-stroke were truly non-stroke cases.
 Class 1 (Stroke): Precision = 0.89
 This indicates that 89% of the cases predicted as stroke were correctly identified.High precision is important because it reduces false positive predictions.

* Recall: Recall measures how well the model identifies actual cases of each class.
Class 0 (No Stroke): Recall = 0.89
The model correctly identified 89% of all non-stroke cases.
Class 1 (Stroke): Recall = 0.93
The model successfully detected 93% of the actual stroke cases.
This is particularly important in healthcare because missing a stroke case (false negative) could have serious consequences. A recall of 0.93 shows that the model is highly effective at identifying patients at risk.

* F1-Score: The F1-score is the harmonic mean of precision and recall, providing a balanced measure of model performance.
Class 0 F1-score: 0.91
Class 1 F1-score: 0.91
The equal F1-scores indicate that the model performs consistently well across both classes.

* Support: Support represents the number of actual instances for each class in the dataset.
Non-stroke cases: 912
Stroke cases: 925
This nearly balanced distribution reflects the effect of the SMOTE technique used earlier to handle class imbalance.

## Why the Decision Tree Model Performed Best

Among the machine learning models tested, the Decision Tree Classifier produced the best overall performance for the stroke prediction task. This can be attributed to several factors related to the nature of the dataset and the strengths of the algorithm.

a) Ability to Handle Mixed Data Types

The stroke dataset contains both numerical variables (such as age, BMI, and glucose level) and categorical variables (such as gender, work type, and smoking status). Decision Trees are well suited for datasets with mixed feature types because they split the data based on feature values without requiring complex transformations.

b) Capturing Non-Linear Relationships

Stroke risk is influenced by multiple interacting factors such as age, hypertension, heart disease, and lifestyle choices. Decision Trees are capable of capturing non-linear relationships and complex feature interactions, which helps improve predictive performance.

c) Improved Performance After Handling Class Imbalance

The dataset originally contained more non-stroke cases than stroke cases, which could bias the model. The use of SMOTE (Synthetic Minority Oversampling Technique) balanced the dataset by generating synthetic examples of the minority class. This allowed the Decision Tree model to learn patterns associated with stroke cases more effectively.

d) Strong Predictive Performance

The Decision Tree model demonstrates strong and balanced predictive performance, with high precision, recall, and F1-scores across both classes. The particularly high recall for stroke cases (0.93) suggests that the model is effective at identifying individuals at risk of stroke, which is crucial in medical prediction tasks.

e) Model Interpretability

Finally,  Decision Trees are easy to interpret and visualize. Healthcare professionals can understand the decision rules used by the model, making it more suitable for real-world medical decision support.

  ## Confusion Matrix Analysis
  

To further evaluate the performance of the Decision Tree model, a confusion matrix was used. The confusion matrix provides a detailed breakdown of the model's predictions by comparing the actual values with the predicted values,confirming that the Decision Tree model provides reliable and balanced predictions for stroke risk classification.

The model correctly identified 857 stroke cases, demonstrating strong ability to detect patients at risk.

The number of false negatives (68) is relatively low, which is important in healthcare applications where missing a stroke case could have serious consequences.
Model Interpretation 
The model also correctly classified a large number of non-stroke cases (811).

## Model Interpretation: Decision Tree Analysis



​To understand how our model predicts stroke risk, we visualised a section of the Decision Tree.
​Key Components of the Tree include 
* ​Root Node (Top): The model identifies Age as the most significant predictor. The first split occurs at 49.001 years.
  
* ​Gini Impurity: This value represents how "mixed" the data is in each node.
  
* ​Gini = 0.5 (Root) indicates a perfect 50/50 split (maximum uncertainty).
  
* ​Gini = 0.184 (Left Orange Node) shows high certainty for the "No Stroke" class.
  
* ​Samples: Represents the number of records (patients) being evaluated at that specific branch.
  
* ​Value [x, y]: Shows the distribution of [No Stroke, Stroke] cases.

## Insights
* The Age is the primary predictor:Patients under 49 are classified with high confidence into the "No Stroke" category (2,178 cases vs 248).

* ​High-Risk Segment:For those over 49 the model digs deeper and further segments older patients at the 67-year mark. Those over 67 (far-right blue node) represent the highest density of stroke cases in this dataset (n=2,253).
 
* ​Model Confidence: The deep blue and orange colours indicate "pure" nodes where the model is very confident in its prediction, while the light blue node (Gini = 0.495) suggests that age alone isn't enough to be certain, and other features (like BMI or Glucose levels) might be needed for a tie-breaker.

# Recommendations
 Clinical & Practical Recommendations
​Targeted Screening Programs: Since the model identifies Age 49 and 67 as significant risk thresholds, healthcare providers should prioritise cardiovascular screenings (blood pressure, cholesterol) for patients entering these age brackets.
​Early Intervention: For the "High-Risk" group (Age 67+), the data suggests a need for more aggressive preventative care or remote monitoring tools to track stroke warning signs.
​Educational Outreach: Public health campaigns could be tailored specifically toward the 50+ demographic, focusing on lifestyle changes that mitigate the risks identified by the model.
​💻 Technical & Model Recommendations
​Feature Expansion: To improve the "Caution" zones where the model is less certain (the light blue nodes), future iterations should include additional variables like Family History, Smoking Status, or Physical Activity levels.
​Address Class Imbalance: If the original dataset had far more "No Stroke" cases than "Stroke" cases, I’d recommend using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns the "Stroke" patterns more effectively.
​Try Ensemble Methods: While the Decision Tree is great for visibility, I recommend testing Random Forests or XGBoost to see if we can boost accuracy while keeping this tree as a "map" for explanation.
