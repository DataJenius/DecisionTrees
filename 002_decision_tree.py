#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause

Run some decision trees to see how well we can classify
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


#############################################################################
# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]



#####################################################################################
# create a numpy array with our y_labels (1 = DO NOT agree w/ White Nationalists)
target_col = 'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'
target_val = 'Agree'
labels = np.zeros(len(df),dtype=np.int)
labels[df.index[df[target_col] != target_val].tolist()]=1



#############################################################################
# design boolean features for use in our decision tree

# try to target the people who *don't* agree with white nationalists
# this is so the tree map colors will make sense

# Age and Income as a continuous features
df["Age"] = df["Age"].fillna(0)
df["Income"] = df["Income"].fillna(0)

# Do you earn $100,000 or more? 1 = Yes
income = np.zeros(len(df),dtype=np.int)
income[df.index[df['Income'] >= 100000].tolist()]=1

# Split political party into three boolean features
independent = np.zeros(len(df),dtype=np.int)
republican = np.zeros(len(df),dtype=np.int)
democrat = np.zeros(len(df),dtype=np.int)
independent[df.index[df['Political Affiliation ']=='Independent'].tolist()]=1
republican[df.index[df['Political Affiliation ']=='Strong Republican'].tolist()]=1
republican[df.index[df['Political Affiliation ']=='Not Strong Republican'].tolist()]=1
democrat[df.index[df['Political Affiliation ']=='Strong Democrat'].tolist()]=1
democrat[df.index[df['Political Affiliation ']=='Not Strong Democrat'].tolist()]=1

# Create boolean feature for not male or female
other_gender = np.zeros(len(df),dtype=np.int)
other_gender[df.index[df['Gender']=='Other'].tolist()]=1
other_gender[df.index[df['Gender']=='DK/REF'].tolist()]=1

# Did you go to college?  1 = Yes
college = np.zeros(len(df),dtype=np.int)
college[df.index[df['What is your highest level of education?']=='Some college'].tolist()]=1
college[df.index[df['What is your highest level of education?']=='College degree'].tolist()]=1
college[df.index[df['What is your highest level of education?']=='Graduate degree'].tolist()]=1

# Split race into four boolean features
white = np.zeros(len(df),dtype=np.int)
latino = np.zeros(len(df),dtype=np.int)
black = np.zeros(len(df),dtype=np.int)
asian = np.zeros(len(df),dtype=np.int)
white[df.index[df['What is your race?']=='White'].tolist()]=1
latino[df.index[df['What is your race?']=='Latino'].tolist()]=1
black[df.index[df['What is your race?']=='Black'].tolist()]=1
asian[df.index[df['What is your race?']=='Asian'].tolist()]=1

# Q1 approve of Trump = 1
q1 = np.zeros(len(df),dtype=np.int)
q1[df.index[df['Do you approve or disapprove of how Donald Trump is handling his job as president?']=='Approve'].tolist()]=1

# Q2 love America = 1
q2 = np.zeros(len(df),dtype=np.int)
q2[df.index[df['Would you say that you love America?']=='Yes'].tolist()]=1

# Q3 help poor = 1
q3 = np.zeros(len(df),dtype=np.int)
q3[df.index[df['Do you think that government policies should help those who are poor and struggling in America?']=='Yes'].tolist()]=1

# Q4 whites racist = 1
q4 = np.zeros(len(df),dtype=np.int)
q4[df.index[df['Do you think that most white people in America are racist?']=='Yes'].tolist()]=1

# Q5 no lost friendships = 1
q5 = np.zeros(len(df),dtype=np.int)
q5[df.index[df['Have you lost any friendships or other relationships as a result of the 2016 presidential election?']=='No'].tolist()]=1

# Q6 civil war likely = 1
q6 = np.zeros(len(df),dtype=np.int)
q6[df.index[df['Do you think it is likely or unlikely that there will be a Civil War in the United States within the next decade?']=='Likely'].tolist()]=1

# Q7 gone hunting = 1
q7 = np.zeros(len(df),dtype=np.int)
q7[df.index[df['Have you ever gone hunting?']=='Yes'].tolist()]=1

# Q8 no kale salad = 1
q8 = np.zeros(len(df),dtype=np.int)
q8[df.index[df['Have you ever eaten a kale salad?']=='No'].tolist()]=1

# Q9 vote for the rock = 1
q9 = np.zeros(len(df),dtype=np.int)
q9[df.index[df['If Dwayne "The Rock" Johnson ran for president as a candidate for your political party, would you vote for him?']=='Yes'].tolist()]=1

# Q10 prefer vader = 1
q10 = np.zeros(len(df),dtype=np.int)
q10[df.index[df['Who would you prefer as president of the United States, Darth Vader or Donald Trump?']=='Prefer Vader'].tolist()]=1

# just to visualize everything
white_nationalists = labels

# Organize everything into the "features" dataframe
# comment out features to try different combinations
features = pd.DataFrame(data={
     'Agree w/ White Nationalists' : white_nationalists,
     #'age' : df["Age"],
     #'income' : df["Income"],
     #'earn100k': income, 
     #'Independent' : independent,
     #'Republican' : republican,
     #'Democrat' : democrat,  
     #'other gender' : other_gender,
     #'college': college,
     #'White' : white,
     #'Latino' : latino,
     #'Black' : black,
     #'Asian' : asian,
     #'(Q1) Approve Trump' : q1,
     #'(Q2) Love America' : q2,
     #'(Q3) Help Poor' : q3,
     #'(Q4) Whites Racist' : q4,
     #'(Q5) No Lost Friendships' : q5, 
     #'(Q6) Civil War Likely' : q6,   
     #'(Q7) Gone Hunting' : q7,  
     #'(Q8) No Kale' : q8,
     #'(Q9) Vote Rock' : q9,
     #'(Q10) Prefer Vader' : q10,     
     })

#99.8% if we add continous variables
#96% without any continuous, using 1000 for training (100%)
#90.5% if we always say "Does Not Agree" 


#############################################################################
# train a decision trees using the features defined above
#model = tree.DecisionTreeClassifier(criterion="entropy", max_features=1, min_samples_split=10)
model = tree.DecisionTreeClassifier(criterion="entropy")
model.fit(features,labels)


#############################################################################
# visualize the decision tree as a PDF map
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                feature_names=features.columns,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf("ex1_college.pdf")


#features_train,labels_train


#Predict Output 
predicted = model.predict(features)
 
# check our accuracy
print(accuracy_score(labels,predicted))
