#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause
"""

# load required libs
import pandas as pd
import numpy as np

# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]


############################################
############################################
### Starting with the white nationalists ###
target_col = 'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'
target_val = 'Agree'

# create y_label = 1 for all people who fit above conditions
y_labels = np.zeros(len(df),dtype=np.int)
y_labels[df.index[df[target_col] == target_val].tolist()]=1

# drop y variable from our features
df = df.drop(columns=[target_col])

# organize our desired data
jojo = df.iloc[:,[0,2]]
jojo = jojo.fillna(0)

jojo["Gender"]=df["Gender"].astype('category')
jojo["Party"]=df["Political Affiliation "].astype('category')
jojo["Approve"]=df["Do you approve or disapprove of how Donald Trump is handling his job as president?"].astype('category')
jojo["Education"]=df["What is your highest level of education?"].astype('category')
jojo["Race"]=df["What is your race?"].astype('category')
jojo["Help_Poor"]=df["Do you think that government policies should help those who are poor and struggling in America?"].astype('category')
jojo["White_Racists"]=df["Do you think that most white people in America are racist?"].astype('category')
# One Hot Encoding
jojo = pd.get_dummies(jojo, columns=["Gender","Party","Approve","Education","Race","Help_Poor","White_Racists"])
print(jojo.head())
print(jojo.dtypes)
cols = jojo.columns




# turn into a matrix for sklearn
#jojo = jojo.as_matrix()

############################################
# run a DT and see what we get
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion="entropy", max_features=1, min_samples_split=10)
model.fit(jojo,y_labels)

#predict = model.predict([[100,35]])
#print(predict)


############################################
# show me
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                feature_names=jojo.columns,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("hack2.pdf")
#Image(graph.create_png())
