#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause
"""

# load required libs
import pandas as pd
import numpy as np
from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge

# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]

# look at income- how poor are they?
# break into poverty, median, quartile(s)?
"""
df["income_level"]='Unknown'

df.loc[df["Income"]>112262,"income_level"] = "5th Quintile"
df.loc[df["Income"]<=112262,"income_level"] = "4th Quintile"
df.loc[df["Income"]<=72001,"income_level"] = "3rd Quintile"
df.loc[df["Income"]<=43511,"income_level"] = "2nd Quintile"
df.loc[df["Income"]<=22800,"income_level"] = "1st Quintile"

df.loc[df["Income"]>22800,"income_level"] = "Above Poverty"
df.loc[df["Income"]<=22800,"income_level"] = "Poverty"


# ignore unknown income
df = df.loc[df["income_level"]!="Unknown"]

print(df.head())


# investigate this df
print("What's in this DF?\n")
for col in df.columns:
    print(col)
    print(df[col].dtypes)    
    print(df[col].unique())
    print("\n")
    
# what variable should we target?
# 95 / 1000 agree with white nationalists    
print(len(df.loc[df['From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?']=='Agree']))

# 139/ 1000 lost friends over the 2016 election    
print(len(df.loc[df['Have you lost any friendships or other relationships as a result of the 2016 presidential election?']=='Yes']))
"""

############################################
############################################
### Starting with the white nationalists ###
target_col = 'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'
target_val = 'Agree'

# create y_label = 1 for all people who "Agree" with "'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'"
#y_labels = np.zeros(len(df),dtype=np.int)
#y_labels[df.index[df[target_col] == target_val].tolist()]=1

# two dataframes - white nationalists and everyone else
wn = df.loc[df[target_col]==target_val]
ee = df.loc[df[target_col]!=target_val]


############################################################
# function to make cool pie graphs with groups
def mojo_pie_chart(df,wn,ee,question):
    """generate a pie chart from Pandas df using Bokeh"""

    # organize our data for bokeh
    d = {'answers':[],'all':[],'wn':[],'ee':[]}
    for answer in df[question].unique():
        d["answers"].append(answer)  
        d["all"].append(len(df.loc[df[question]==answer])/len(df))        
        d["wn"].append(len(wn.loc[wn[question]==answer])/len(wn))        
        d["ee"].append(len(ee.loc[ee[question]==answer])/len(ee))        

    # get max y value
    max_y = max(d["wn"]+d["ee"]+d["all"])

    # plot via bokeh
    source = ColumnDataSource(data=d)

    p = figure(x_range=d["answers"], y_range=(0, max_y*1.25), plot_height=350, title=question,
               toolbar_location=None, tools="")

    p.vbar(x=dodge('answers', -0.25, range=p.x_range), top='all', width=0.2, source=source,
           color="#477dd3", legend=value("All Respondents"))
    
    p.vbar(x=dodge('answers', 0, range=p.x_range), top='wn', width=0.2, source=source,
           color="#e54a1b", legend=value("Agree w/ White Nationalists"))

    p.vbar(x=dodge('answers', 0.25, range=p.x_range), top='ee', width=0.2, source=source,
           color="#0d9129", legend=value("Everyone Else"))
        
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    show(p)
    
    
# generate a pie chart using our function
#mojo_pie_chart(df,wn,ee,'Do you approve or disapprove of how Donald Trump is handling his job as president?')    
#mojo_pie_chart(df,wn,ee,'What is your highest level of education?')    
#mojo_pie_chart(df,wn,ee,'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?')    
#mojo_pie_chart(df,wn,ee,'Would you say that you love America?')    

# this one is baffling...
#mojo_pie_chart(df,wn,ee,'What is your race?')        

# 117 latinos, 16 are pro WN
#len(df.loc[df["What is your race?"]=="Latino"])
#len(df.loc[(df["What is your race?"]=="Latino") & (df[target_col]==target_val)])

# 50 asians, 8 are pro WN
#len(df.loc[df["What is your race?"]=="Asian"])
#len(df.loc[(df["What is your race?"]=="Asian") & (df[target_col]==target_val)])

# 135 blacks, 12 are pro WN
#len(df.loc[df["What is your race?"]=="Black"])
#len(df.loc[(df["What is your race?"]=="Black") & (df[target_col]==target_val)])

# moar graphs
#mojo_pie_chart(df,wn,ee,'Do you think that government policies should help those who are poor and struggling in America?') 
#mojo_pie_chart(df,wn,ee,'Do you think that most white people in America are racist?') 
#mojo_pie_chart(df,wn,ee,'Have you lost any friendships or other relationships as a result of the 2016 presidential election?') 

# messy graph but interesting
#mojo_pie_chart(df,wn,ee,'Political Affiliation ') 

# they expect a civil war is coming
#mojo_pie_chart(df,wn,ee,'Do you think it is likely or unlikely that there will be a Civil War in the United States within the next decade?') 
#mojo_pie_chart(df,wn,ee,'Have you ever gone hunting?') 
    
# no more likely to go hunting, but way less likely to eat salad    
#mojo_pie_chart(df,wn,ee,'Have you ever eaten a kale salad?') 

#mojo_pie_chart(df,wn,ee,'If Dwayne "The Rock" Johnson ran for president as a candidate for your political party, would you vote for him?') 
#mojo_pie_chart(df,wn,ee,'Who would you prefer as president of the United States, Darth Vader or Donald Trump?') 

#mojo_pie_chart(df,wn,ee,'income_level') 
mojo_pie_chart(df,wn,ee,'Gender') 

# https://bokeh.pydata.org/en/latest/docs/gallery/bar_stacked.html



# use information gain to find the most divisive topics

"""
# overall how many people would prefer Trump to Vader?
# 490 / 1000
len(df.index[df['Who would you prefer as president of the United States, Darth Vader or Donald Trump?'] == "Donald Trump"].tolist())


df["y"]=0
y_index = df.loc[df['From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?']=='Agree']
print(y_index)
#print(df.head())



# what is our starting entropy
prob_a = 95/1000
prob_b = 1 - prob_a
entropy = -prob_a*log(prob_a) - prob_b*log(prob_b)
print(prob_a)
print(prob_b)
scipy.stats.entropy()
#Entropy = - p(a)*log(p(a)) - p(b)*log(p(b))



import scipy.stats.entropy

jojo = df.iloc(:,df[10]=='Yes')

jojo = df.iloc[10=='Agree',:]


jojo = df.loc[df['From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?']=='Agree']
"""