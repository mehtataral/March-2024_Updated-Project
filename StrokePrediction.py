# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:55:53 2023

@author: Livewire
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"E:\Taral\DataScience\Project\healthcare-dataset-stroke-data.csv")
print(df1)
# stroke: 1 if the patient had a stroke or 0 if not
# bmi: body mass index
# avg_glucose_level: average glucose level in blood

df1.isnull().sum()
df1.dropna(inplace =True)
df1

df1.drop(columns = "id",inplace =True)
df1.drop(columns = "r",inplace =True)

df1

df1.nlargest(5,"bmi")
df1.nsmallest(5,"bmi")

# The World Health Organization (WHO) has established the following BMI categories for adults:

# Underweight: BMI less than 18.5
# Normal weight: BMI 18.5 to 24.9
# Overweight: BMI 25 to 29.9
# Obesity Class I: BMI 30 to 34.9
# Obesity Class II: BMI 35 to 39.9
# Obesity Class III: BMI 40 or higher

## 1.
Normal_weight1=df1.bmi>=18.5 
Normal_weight2=df1.bmi<=24.9 

df1.loc[Normal_weight1 & Normal_weight2,"Remark"] ="Normal_weight"
df1

## 2.
Underweight=df1.bmi<18.5 
df1.loc[Underweight,"Remark"] ="Underweight"
df1

## 3.
Overweight1=df1.bmi>=25 
Overweight2=df1.bmi<=29.9
df1.loc[Overweight1 & Overweight2,"Remark"] ="Overweight"
df1

## 4.
Obesity_Class_I1=df1.bmi>=30
Obesity_Class_I2=df1.bmi<=34.9
df1.loc[Obesity_Class_I1 & Obesity_Class_I2,"Remark"] ="Obesity Class I"
df1

## 5.
Obesity_Class_II1=df1.bmi>=35
Obesity_Class_II2=df1.bmi<=39.9
df1.loc[Obesity_Class_II1 & Obesity_Class_II2,"Remark"] ="Obesity Class II"
df1

## 6.
Obesity_Class_III=df1.bmi>=40 
df1.loc[Obesity_Class_III,"Remark"] ="Obesity Class III"
df1


df1.Remark.unique()
df1.Remark.nunique()

grp = df1.groupby("Remark").count()
grp.index
grp.gender

plt.figure(figsize=(10,12))
sns.displot(df1['age'])
# This will plot a distribution plot of variable age


plt.figure(figsize=(10,12))
ax = sns.barplot(data = grp,x = grp.index,y = grp.gender)
plt.title("Count of gender with Body Weight Type with stroke ")
for i in ax.containers:
     ax.bar_label(i)

#--------------------------------------------------
stroke_yes = df1.loc[(df1.stroke == 1)] #stroke  hai
stroke_yes

grp2  = stroke_yes.groupby("Remark").count().reset_index()
grp2


plt.figure(figsize=(12,7))
plt.title("Body Weight Type in Percentage wise with stroke")
plt.pie(grp2.gender,labels =grp2.Remark,autopct="%1.1f%%")
plt.legend()

stroke_yes.nlargest(5,"age")
stroke_yes.nsmallest(5,"age")

stroke_yes.age=stroke_yes.age.astype(int)
stroke_yes

stroke_yes.nlargest(5,"age")
stroke_yes.nsmallest(5,"age")

stroke_yes.loc[stroke_yes.age <=19,"Group"]="Below 19"
stroke_yes

stroke_yes.loc[(stroke_yes.age >=20)&(stroke_yes.age <=45) ,"Group"]="20-45 Years"
stroke_yes

stroke_yes.loc[(stroke_yes.age >=46)&(stroke_yes.age <=65) ,"Group"]="46-65 Years"
stroke_yes

stroke_yes.loc[(stroke_yes.age >=66),"Group"]="66+ Years"
stroke_yes

plt.figure(figsize=(12,7))
plt.title("Count of Age group with stroke")
sns.countplot(data = stroke_yes,x ="Group",hue ="gender")


plt.figure(figsize=(12,7))
plt.title("Count of working professional with stroke")
sns.countplot(data = stroke_yes,x ="work_type",hue ="gender")

plt.figure(figsize=(12,7))
plt.title("Count of Smoking Status with stroke")
sns.countplot(data = stroke_yes,x ="smoking_status",hue ="gender")

plt.figure(figsize=(12,7))
plt.title("Count of Residence_type with stroke")
sns.countplot(data = stroke_yes,x ="Residence_type",hue ="gender")


plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with Heart Probelm getting stroke")
sns.countplot(data = stroke_yes,x ="Remark",hue ="heart_disease")

# ----------------------------------------

female_yes = stroke_yes.loc[stroke_yes.gender =="Female"]
female_yes

plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with stroke in Female")
sns.countplot(data = stroke_yes,x ="Remark",hue ="work_type")


plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with stroke in Female")
ax = sns.barplot(data= female_yes, x = "Remark",y ="bmi")

female_yes.info()
for x in ax.containers:
    ax.bar_label(x)


plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with age group in Female stroke")
sns.countplot(data = female_yes,x ="Remark",hue ="Group")

plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with ever_married in Female stroke")
sns.countplot(data = female_yes,x ="Remark",hue ="ever_married")


plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with avg_glucose_level in Female stroke")
sns.barplot(data =female_yes, x= "Remark", y="avg_glucose_level")

plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with heart_disease in Female stroke")
sns.barplot(data =female_yes, x= "Remark", y="avg_glucose_level",hue ="heart_disease")

#----------------------------------------------------------

male_yes = stroke_yes.loc[stroke_yes.gender =="Male"]
male_yes

plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with work_type in male stroke")
sns.countplot(data = male_yes,x ="Remark",hue ="work_type")

plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with bmi in male stroke")
ax = sns.barplot(data= male_yes, x = "Remark",y ="bmi")
male_yes.info()

for x in ax.containers:
    ax.bar_label(x)


plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with age group in male stroke")
sns.countplot(data = male_yes,x ="Remark",hue ="Group")
plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with ever_married in male stroke")
sns.countplot(data = male_yes,x ="Remark",hue ="ever_married")


plt.figure(figsize=(12,7))
plt.title("Count of Body Mass Index with avg_glucose_level in male stroke")
sns.barplot(data =male_yes, x= "Remark", y="avg_glucose_level",hue ="heart_disease")

# -----------------------------------
stroke_no = df1.loc[(df1.stroke == 0)] #stroke nhi hai
stroke_no

grp3  = stroke_no.groupby("Remark").count().reset_index()
grp3

plt.pie(grp3.gender,labels =grp2.Remark,autopct="%1.1f%%")

# stroke_no.loc[stroke_no.age <=19,"Group"]="Below 19"
# stroke_no

# stroke_no.loc[(stroke_no.age >=20)&(stroke_no.age <=45) ,"Group"]="20-45 Years"
# stroke_no

# stroke_no.loc[(stroke_no.age >=46)&(stroke_no.age <=65) ,"Group"]="46-65 Years"
# stroke_no

# stroke_no.loc[(stroke_no.age >=66),"Group"]="66+ Years"
# # stroke_no

# stroke_no
# def glucose_status(value):
#     if value <=70:
#         print("value low",value)
#         return "Low"
#     elif value >70 and value <=126:
#         print("value normal",value)
#         return "Normal"
#     elif value >126 and value <=182:
#         print("value broder line",value)
#         return "Broder Line"
#     elif value >182 and value <=250:
#         print("value high",value)
#         return "high"
#     else:
#         print("value dangerous",value)
#         return "dangerous"

# stroke_no['glucose_status']  = stroke_no["avg_glucose_level"].map(glucose_status)

# -----------------------------------------
# import pandas as pd

# # Create a DataFrame
# data = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10]})

# # Define a mapping
# mapping = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

# # Apply the mapping using map() to a specific column
# data['A'] = data['A'].map(mapping)

# print(data)


# plt.figure(figsize=(12,7))
# plt.title("Count of Age group without stroke")
# sns.countplot(data = stroke_no,x ="Group",hue ="gender")


# plt.figure(figsize=(12,7))
# plt.title("Count of working professional without stroke")
# sns.countplot(data = stroke_no,x ="work_type",hue ="gender")

# plt.figure(figsize=(12,7))
# plt.title("Count of Smoking Status without stroke")
# sns.countplot(data = stroke_no,x ="smoking_status",hue ="gender")

# plt.figure(figsize=(12,7))
# plt.title("Count of Residence_type without stroke")
# sns.countplot(data = stroke_no,x ="Residence_type",hue ="gender")


# plt.figure(figsize=(12,7))
# plt.title("Count of Body Mass Index with Heart Probelm getting without stroke")
# sns.countplot(data = stroke_no,x ="Remark",hue ="heart_disease")


#---------------------------------------------------

# Average Glucose Level

# Normal      <117
# Border      117 - 137
# Diabetic    >= 137
stroke_yes.avg_glucose_level
x1 =stroke_yes.nlargest(5,"avg_glucose_level")
x1.avg_glucose_level

x2 =stroke_yes.nsmallest(5,"avg_glucose_level")
x2.avg_glucose_level

stroke_yes.loc[stroke_yes.avg_glucose_level <117,"BSL"] ="Normal"
stroke_yes

stroke_yes.loc[(stroke_yes.avg_glucose_level >= 117)&(stroke_yes.avg_glucose_level <137),"BSL"] ="Border"
stroke_yes

stroke_yes.loc[stroke_yes.avg_glucose_level >137,"BSL"] ="Diabetic"
stroke_yes

stroke_yes.BSL.unique()

plt.figure(figsize=(12,7))
sns.countplot(data =stroke_yes,x ="BSL",hue ="gender")

plt.figure(figsize=(12,7))
sns.countplot(data =stroke_yes,x = "BSL",hue = "Group")

plt.figure(figsize=(12,7))
sns.countplot(data =stroke_yes,x = "BSL",hue = "Remark")


plt.figure(figsize=(12,7))
sns.barplot(data =stroke_yes,x = "BSL",y = "avg_glucose_level",hue = "Group")


#--------------------------------Prediction----------------------------

features = stroke_yes[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
features

#let's do one hot encoding on our numerical values
features_one_hot = pd.get_dummies(features, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

stroke_yes.info()


df1.loc[df1.avg_glucose_level <117,"BSL"] ="Normal"
df1

df1.loc[(df1.avg_glucose_level >= 117)&(df1.avg_glucose_level <137),"BSL"] ="Border"
df1

df1.loc[df1.avg_glucose_level >137,"BSL"] ="Diabetic"
df1

df1.BSL.unique()


df1.loc[df1.age <=19,"Group"]="Below 19"
df1

df1.loc[(df1.age >=20)&(df1.age <=45) ,"Group"]="20-45 Years"
df1

df1.loc[(df1.age >=46)&(df1.age <=65) ,"Group"]="46-65 Years"
df1

df1.loc[(df1.age >=66),"Group"]="66+ Years"
df1

df1.Group.unique()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1.gender = le.fit_transform(df1.gender)
df1.gender

df1.ever_married = le.fit_transform(df1.ever_married)
df1.ever_married

df1.work_type = le.fit_transform(df1.work_type)
df1.work_type

df1.Residence_type = le.fit_transform(df1.Residence_type)
df1.Residence_type

df1.smoking_status = le.fit_transform(df1.smoking_status)
df1.smoking_status

df1.Remark = le.fit_transform(df1.Remark)
df1.Remark

df1.Group = le.fit_transform(df1.Group)
df1.Group

df1.BSL = le.fit_transform(df1.BSL)
df1.BSL

df1.info()
df1


#----------------------------------------------------------------

df1= df1[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status','Remark', 'BSL', 'Group','stroke']]

df1

sns.boxplot(x='stroke',y='age',data=df1,palette='muted')
plt.show()


sns.boxplot(x='stroke',y='avg_glucose_level',data=df1,palette='muted')
plt.show()

sns.boxplot(x='stroke',y='bmi',data=df1,palette='muted')
plt.show()

sns.heatmap(df1.corr(), cmap="YlGnBu", annot=False)

x =df1.iloc[:,:-1]
x

y = df1.iloc[:,-1]
y
from sklearn.feature_selection import SelectKBest, f_classif

classifier = SelectKBest(score_func=f_classif,k=5)
fits = classifier.fit(df1.drop('stroke',axis=1),df1['stroke'])
x1=pd.DataFrame(fits.scores_)
columns = pd.DataFrame(df1.drop('stroke',axis=1).columns)
fscores = pd.concat([columns,x1],axis=1)
fscores.columns = ['Attribute','Score']
fscores.sort_values(by='Score',ascending=False)

cols=fscores[fscores['Score']>50]
print(cols)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=44)

x_train
x_test
y_train
y_test


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

# max_depth: This hyperparameter specifies the maximum depth of 
# each decision tree in the random forest. 
# A smaller value of max_depth can prevent overfitting by limiting 
# the complexity of individual trees. In your case, max_depth=2 
# restricts each tree to have a maximum depth of 2 levels.


# This hyperparameter sets the random seed for reproducibility. 
# When you set a specific random_state value, the random number 
# generation process will be the same every time you run the code.
# This helps ensure that the results are consistent across different runs.


# The confusion matrix is a table used in classification to summarize
# the performance of a classification model on a set of data 
# for which the true values are known.

# True Positives (TP): The number of instances that were correctly predicted as positive.
# True Negatives (TN): The number of instances that were correctly predicted as negative.
# False Positives (FP): The number of instances that were incorrectly predicted as positive when they are actually negative (Type I error).
# False Negatives (FN): The number of instances that were incorrectly predicted as negative when they are actually positive (Type II error).

# The output will look something like this:
    
# [[2 1]
#  [1 4]]    
    
# The first row corresponds to the actual negative class (0).
# The first column corresponds to the predicted negative class (0).
# The second row corresponds to the actual positive class (1).
# The second column corresponds to the predicted positive class (1).

# There are 2 true negatives (TN).
# There is 1 false positive (FP).
# There is 1 false negative (FN).
# There are 4 true positives (TP).
    
# You can use the confusion matrix to calculate various evaluation 
# metrics such as accuracy, precision, recall, F1-score, and more, which provide deeper insights into the performance of your classification model.

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, y_pred))

# classification_report is a function provided by the 
# sklearn.metrics module in scikit-learn, 
# a popular machine learning library in Python.
# It's used to generate a text report that provides a comprehensive overview of
# the performance of a classification model.

# This report includes various metrics such as 
# precision, recall, F1-score, and support 
# for each class in the classification problem.

# Precision: Indicates how many of the predicted positive instances were actually positive.
# Recall: Indicates how many of the actual positive instances were correctly predicted as positive.
# F1-score: The harmonic mean of precision and recall, which provides a balanced measure between them.
# Support: The number of actual occurrences of each class in the data.
# Accuracy: The overall accuracy of the model's predictions.
# Macro avg: The average of metrics calculated for each class independently.
# Weighted avg: The average of metrics weighted by the number of true instances for each class.

# classification_report is a helpful tool for understanding how well your classification model is performing for each class and across different metrics. It's especially useful when dealing with imbalanced datasets or multi-class classification problems.
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))

clf.get_params

params={
    "n_estimators":[90,100,115,130],
    "criterion":['gini','entropy'],
    "max_depth":range(2,20,1),
    "min_samples_leaf":range(1,10,1),
    "min_samples_split":range(2,10,1),
    "max_features":['auto','log2']
}
from sklearn.model_selection import GridSearchCV
# cv is the number of cross-validation folds used to evaluate each combination of hyperparameters.

grid_svc = GridSearchCV(estimator=clf, param_grid=params,cv=2,n_jobs=-1,verbose=3)
grid_svc


# grid_svc.fit(x_train, y_train)
# grid_svc.best_params_

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

classifiers=[['Naive Bayes :', GaussianNB()],['LogisticRegression :', LogisticRegression(max_iter = 1000)], ['DecisionTree :',DecisionTreeClassifier()]]
classifiers

for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(x_train, y_train.ravel())
    predictions = classifier.predict(x_test)
    print(name, accuracy_score(y_test, predictions))

from sklearn.ensemble import VotingClassifier
clf1 = GaussianNB()
clf2 = DecisionTreeClassifier()
clf3 = LogisticRegression()

# Voting ensembles are the ensemble machine learning technique,
# one of the top-performing models among all machine learning 
# algorithms. As voting ensembles are the most used ensemble techniques

# What is Voting Ensemble? How does it work?
# Voting ensembles are the type of machine learning algorithms
# that falls under the ensemble techniques. 
# As they are one of the ensemble algorithms,
# they use multiple models to train on the dataset and for predictions.

# They are two categories of voting ensembles.

# Classification
# Regression

# Here, the category most predicted by the multiple algorithms will be treated as the final prediction of the model.
# For Example, if three models predict YES and two models predict NO, YES would be considered the final prediction of the model.

# Voting Regressors are the same as voting classifiers. Still, they are used on regression problems, and the final output from this model is the mean of the prediction of all individual models.

# For Example, if the outputs from the three models are 5,10, and 15, then the final result would be the mean of these values, which is 15.

vot_hard = VotingClassifier(estimators= classifiers, voting='hard')
vot_hard.fit(x_train, y_train)

vot_soft = VotingClassifier(estimators = classifiers, voting ='soft')
vot_soft.fit(x_train, y_train)

print("Training data accuracy:", vot_hard.score(x_train, y_train))
print("Testing data accuracy", vot_hard.score(x_test, y_test))

print("Training data accuracy:", vot_soft.score(x_train, y_train))
print("Testing data accuracy", vot_soft.score(x_test, y_test))

# https://github.com/akpythonyt/ML-projects/blob/main/brain-stroke-prediction-with-less-visualizations.ipynb
# https://www.analyticsvidhya.com/blog/2021/05/how-to-create-a-stroke-prediction-model/
# https://www.kaggle.com/code/shaniquehines/stroke-analysis-and-prediction#Model-Evaluation
# https://www.kaggle.com/code/madhurajoshi98/stroke-prediction-and-analysis



