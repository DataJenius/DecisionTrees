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
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import NumeralTickFormatter, Range1d
import math

#############################################################################
# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]


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
print(features.head())





#############################################################################
# train a decision trees using the features defined above
#model = tree.DecisionTreeClassifier(criterion="entropy", max_features=1, min_samples_split=10, random_state=42)
model = tree.DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(features,labels)

#Predict Output 
predicted = model.predict(features)
 
# check our accuracy
print(accuracy_score(labels,predicted))


# look at feature importance
feature_importance = {}
for idx, val in enumerate(model.feature_importances_):
    key = features.columns[idx]
    feature_importance[idx]=[val,key]

# sort and make readable
feature_importance = pd.DataFrame.from_dict(feature_importance, orient='index').rename(columns={0:'Information-Gain',1:'Feature'})
feature_importance = feature_importance.sort_values(by=['Information-Gain'],ascending=False)
#print(feature_importance)


#############################################################################
# Visualize via Bokeh

 # put data in Bokeh format    
source = ColumnDataSource(
        data = {'Feature' : feature_importance["Feature"],
                'InfoGain' : feature_importance["Information-Gain"]})
    
# use data for tooltips
hover = HoverTool(
        tooltips=[
                ("Feature", "@Feature"), 
                ("Information Gain", "@InfoGain"),                 
            ]
        )    

# plot our information gain
p1 = figure(x_range=feature_importance["Feature"].unique(), plot_width=800, title="Information Gain", tools=['pan','box_zoom','reset',hover])
p1.vbar(x='Feature', top='InfoGain', width=0.9, source=source)
p1.xaxis.axis_label = "Feature"
p1.yaxis.axis_label = "Information Gain"
p1.yaxis.formatter=NumeralTickFormatter(format="0.00000")
p1.xaxis.major_label_orientation = math.pi/2
p1.y_range=Range1d(0, .4)
#tab1 = Panel(child=p1, title='Information Gain')
show(p1)


