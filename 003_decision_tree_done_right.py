#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause

Use best practices to create a proper generalized decision tree model
"""

#############################################################################
# load required libs
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# visualize our tree
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus



#############################################################################
# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]

# shuffle up our data set
df = df.sample(frac=1, random_state=42)



#####################################################################################
# Create sklearn-friendly features and labels

# create a numpy array with our labels (1 = agree w/ White Nationalists)
# then remove this column from our dataframe so it does not become a feature
target_col = 'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'
target_val = 'Agree'
labels = np.zeros(len(df),dtype=np.int)
labels[df.index[df[target_col] == target_val].tolist()]=1
df = df.drop([target_col], axis=1)

# clean up these goofy column names
clean_cols = ['income','gender','age','party','Q1','education','race','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
df.columns = clean_cols

# treat any NaN or missing continuous values as 0
df["income"] = df["income"].fillna(0)
df["age"] = df["age"].fillna(0)

# clean up all of our categorical features
cat_features = clean_cols[:]
cat_features.remove('income')
cat_features.remove('age')
for cat in cat_features:
    df.loc[df[cat]=='DK/REF',cat] = "NA" 

# trying to make the resulting visualizations more readable
df.loc[df['party']=='Strong Republican','party'] = "Republican+"
df.loc[df['party']=='Not Strong Republican','party'] = "Republican-"
df.loc[df['party']=='Strong Democrat','party'] = "Democrat+"
df.loc[df['party']=='Not Strong Democrat','party'] = "Democrat-"
df.loc[df['education']=='College degree','education'] = "College_degree"
df.loc[df['education']=='Some college','education'] = "Some_college"
df.loc[df['education']=='Graduate degree','education'] = "Grad_degree"
df.loc[df['education']=='High school','education'] = "High_School"
df.loc[df['Q10']=='Donald Trump','Q10'] = "Trump"
df.loc[df['Q10']=='Darth Vader','Q10'] = "Vader"

# break into booleans wit one hot encoding - 54 resulting features
features = pd.get_dummies(df, columns=cat_features)
#print(features.head())



#############################################################################
# start with simple data spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)



#############################################################################
# use GridSearchCV to find optimum parameters
 
# set parameters to try

parameters = {
              'criterion':['entropy','gini'], 
              'class_weight':[{1:1,0:1},{1:2,0:1},{1:1,0:2}],
              'max_features':[None,5,10,15,20,50],
              'max_leaf_nodes':[None,10,20,30,40,50],
              'max_depth':[None,10,20,30,40,50],
              'min_samples_split':[2,10,20,30,40,50,60,70,80,90,100]}
 
# this example uses an tree model
model = tree.DecisionTreeClassifier()
clf = GridSearchCV(model, parameters)
clf.fit(X_train,y_train)
 
# get best fit parameters
print(clf.best_params_)

#Predict Output 
predicted = clf.predict(X_test)

# check our accuracy
print("Accuracy:",accuracy_score(y_test,predicted))
 
# % of all TRUE items that were classified as TRUE
print("Recall score:", recall_score(y_test, predicted))
 
# % of items classified TRUE that are really TRUE
print("Precision score:", precision_score(y_test, predicted))
 
# f1 score is weighted combination of the above
print("F1 Score:",f1_score(y_test, predicted, average='macro'))

tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
print("False Positives:",fp)
print("False Negatives:",fn)




