#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause

Run some decision trees to see how well we can classify

Using ALL data just as a teaching example
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
# create a numpy array with our y_labels (1 = agree w/ White Nationalists)
target_col = 'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'
target_val = 'Agree'
labels = np.zeros(len(df),dtype=np.int)
labels[df.index[df[target_col] == target_val].tolist()]=1


#############################################################################
# design boolean features for use in our decision tree

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
     #'Feature X' : white_nationalists,
     'age' : df["Age"],
     'income' : df["Income"],  # use one or the other
     #'earn100k': income,       # these are redundant features
     'Independent' : independent,
     'Republican' : republican,
     'Democrat' : democrat,  
     'other gender' : other_gender,
     'college': college,
     'White' : white,
     'Latino' : latino,
     'Black' : black,
     'Asian' : asian,
     '(Q1) Approve Trump' : q1,
     '(Q2) Love America' : q2,
     '(Q3) Help Poor' : q3,
     '(Q4) Whites Racist' : q4,
     '(Q5) No Lost Friendships' : q5, 
     '(Q6) Civil War Likely' : q6,   
     '(Q7) Gone Hunting' : q7,  
     '(Q8) No Kale' : q8,
     '(Q9) Vote Rock' : q9,
     '(Q10) Prefer Vader' : q10,     
     })


cumulative_accuracy = [.907,.935,.941,.951,.961,.968,.975,.978,.986,.993,.997,.998,.998,.998,.998,.998,.998,.998,.998,.998,.998]
#99.8% if we add continuous variables, using 1000 for training (100%)
#96% without any continuous, using 1000 for training (100%)
#90.5% if we always say "Does Not Agree" 


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
tab1 = Panel(child=p1, title='Information Gain')
#show(p1)


#############################################################################
# Visualize via Bokeh

 # put data in Bokeh format    
source2 = ColumnDataSource(
        data = {'Feature' : feature_importance["Feature"],
                'Accuracy' : cumulative_accuracy})
    
# use data for tooltips
hover2 = HoverTool(
        tooltips=[
                ("Feature", "Feature"),
            ]
        )    

# plot our cumulative accuracy
p2 = figure(x_range=feature_importance["Feature"].unique(),plot_width=800, title="Cumulative Accuracy", tools=['pan','box_zoom','reset', hover2])
p2.vbar(x='Feature', top='Accuracy', width=0.9, source=source2)
p2.xaxis.axis_label = "Feature"
p2.yaxis.axis_label = "Cumulative Accuracy"
p2.xaxis.major_label_orientation = math.pi/2
p2.yaxis.formatter=NumeralTickFormatter(format="0.00%")
p2.y_range=Range1d(.9, 1)
tab2 = Panel(child=p2, title='Cumulative Accuracy')
#show(p2)

# show as tabs
layout = Tabs(tabs=[tab1, tab2])
output_file('tabs.html')
show(layout)




#############################################################################
# custom function to set color and alpha of each node in the tree
def get_node_color(a,b):
    colorA = "#346ac1"
    colorB = "#c13633"
    if(a > b): 
        color = colorA
        per = a/(a+b)
    else: 
        color = colorB
        per = b/(a+b)
    opacity = int(round((per-.5)/.5,2)*100)
    if(opacity==100):
        opacity=99  
    else: 
        opacity=50        
    return str(color)+str(opacity)


#############################################################################
# visualize the decision tree as a PDF / PNG map
dot_data = StringIO()
export_graphviz(model, 
                out_file=dot_data,  
                feature_names=features.columns,
                filled=True, 
                rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# set custom colors for our leafs
nodes = graph.get_node_list()
for node in nodes:
    if node.get_name() not in ('node', 'edge'):
        values = model.tree_.value[int(node.get_name())][0]
        node.set_fillcolor(get_node_color(values[0],values[1]))        

# output tree as visual         
graph.write_png('full_tree.png')
graph.write_pdf("full_tree.pdf")


# gets two wrong
for idx, val in enumerate(predicted):
    if(val != labels[idx]):
        print(idx, val, labels[idx])

df.iloc[614]
df.iloc[753]
