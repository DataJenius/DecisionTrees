#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:11:42 2018

@author: Josh Pause

Start with some EDA of the data. Try to visualize the feature splits that will 
result in the largest information gain when running via decision tree
"""


#############################################################################
# load required libs
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import NumeralTickFormatter
from bokeh.models.widgets import Panel, Tabs


#############################################################################
# import our csv into a dataframe
df = pd.read_csv('data/201710-CAH_PulseOfTheNation.csv')

# keep only desired columns
df = df.iloc[:,[0,1,2,4,5,6,8,10,13,16,18,20,21,22,23,24,25]]

# make a categorical column for those who "Agree" with "From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?"
target_col = 'From what you have heard or seen, do you mostly agree or mostly disagree with the beliefs of White Nationalists?'
target_val = 'Agree'
df["WN"] = "Do Not Agree"
df.loc[df[target_col] == target_val,"WN"] = "Agree with White Nationalism"

# sort so legends are in correct order
df = df.sort_values(by=['WN'])
#print(df.head())



"""

this code works- don't fuck with it

#############################################################################
# visualize income v age via bokeh for the website

# format our points based on group
colormap = {'Do Not Agree': 'blue', 'Agree with White Nationalism': 'red'}
colors = [colormap[x] for x in df['WN']]

sizemap = {'Do Not Agree': 5, 'Agree with White Nationalism': 10}
sizes = [sizemap[x] for x in df['WN']]

# format our data from the dataframe
source = ColumnDataSource(
        data=dict(
            x=df["Age"],
            y=df["Income"],
            desc=df["WN"],
            colors=colors,
            sizes=sizes
        )
    )

# format our hover tooltips
hover = HoverTool(
        tooltips=[
            ("Age", "@x"),
            ("Income", "@y{$0,0.00}"),            
            ("Opinion", "@desc"),
        ]
    )

# create our figure
p1 = figure(title="Income versus Age", width=800, tools=['pan','box_zoom','reset',hover])

# format our axes
p1.xaxis.axis_label = 'Age'
p1.yaxis.axis_label = 'Annual Income'
p1.yaxis.formatter=NumeralTickFormatter(format="$0,0.00")

# plot our scatter
p1.circle('x', 'y', color='colors', size='sizes', alpha=0.5, legend='desc', source=source)
output_file("eda1.html", title="Income v Age")
show(p1)
"""




#############################################################################
# given a categorical (discrete) variable 
# show % in each answer who agree w/ WN
def plot_discrete_feature(df, discrete):
    # get the answers to our question
    answers = df[discrete].unique().tolist()
    wngroup = df["WN"].unique().tolist()
    wngroup = ['Agree','Disagree']
    # get the relative percentage of each group
    stats_agree = []
    stats_noagree = []
    totals = []
    wn_totals = []
    nwn_totals = []    
    for a in answers:
        # how many people answered this way?
        total = len(df.loc[df[discrete] == a])
        
        # how many people who answered this way also agree w/ WN?
        wn_total = len(df.loc[(df[discrete] == a) & (df["WN"] == 'Agree with White Nationalism')])
        
        # what percentage of people who answered this way agree w/ WN?
        wn_per = wn_total/total
        
        # hold our stats for bokeh format
        stats_agree.append(wn_per)
        stats_noagree.append(1-wn_per)
        totals.append(total)
        wn_totals.append(wn_total) 
        nwn_totals.append(total-wn_total)         

    # put data in Bokeh format    
    source = ColumnDataSource(
        data = {'answers' : answers,
                'Agree' : stats_agree,
                'Disagree' : stats_noagree,
                'totals' : totals,
                'wntotals' : wn_totals,
                'nwntotals' : nwn_totals,})

    # use data for tooltips
    hover = HoverTool(
            tooltips=[
                ("Group", "@totals people said \"@answers\""), 
                ("Agree with White Nationalism", "@Agree{0.00%} (@wntotals people)"), 
                ("Do Not Agree", "@Disagree{0.00%} (@nwntotals people)")           
            ]
        )
    
    # create figure
    p = figure(x_range=answers, width=800, title=discrete, tools=['pan','box_zoom','reset',hover])
    
    # plot stacked bars
    p.vbar_stack(wngroup, x='answers', width=0.9, color=["#c13633","#346ac1"], source=source,
                             legend=['Agree with White Nationalism','Do Not Agree'], name=wngroup)
    
    # format our axes
    p.xaxis.axis_label = discrete
    p.yaxis.axis_label = 'Percent of Respondents who Agree with White Nationalism'
    p.yaxis.formatter=NumeralTickFormatter(format="0.00%")
    
    # output our file
    #output_file("bar_stacked.html")
    #show(p)
    return(p)



#############################################################################
# split income into 3 groups**
    
"""
df["Do you earn $100,000 or more per year?"] = "No Answer"
df.loc[df['Income'] < 100000,"Do you earn $100,000 or more per year?"] = "No"
df.loc[df['Income'] >= 100000,"Do you earn $100,000 or more per year?"] = "Yes"
p1 = plot_discrete_feature(df, 'Do you earn $100,000 or more per year?')
tab1 = Panel(child=p1, title='Income')

# group political affiliation into 4 groups
df["What is your political affiliation?"] = "No Answer"
df.loc[df['Political Affiliation ']=='Independent',"What is your political affiliation?"] = "Independent"
df.loc[df['Political Affiliation ']=='Strong Republican',"What is your political affiliation?"] = "Republican"
df.loc[df['Political Affiliation ']=='Not Strong Republican',"What is your political affiliation?"] = "Republican"
df.loc[df['Political Affiliation ']=='Strong Democrat',"What is your political affiliation?"] = "Democrat"
df.loc[df['Political Affiliation ']=='Not Strong Democrat',"What is your political affiliation?"] = "Democrat"
p2 = plot_discrete_feature(df, 'What is your political affiliation?')
tab2 = Panel(child=p2, title='Party')

# Gender
df.loc[df['Gender']=='DK/REF',"Gender"] = "No Answer"
p3 = plot_discrete_feature(df, 'Gender')
tab3 = Panel(child=p3, title='Gender')


# education***
# should we sort or group further?
df.loc[df['What is your highest level of education?']=='DK/REF',"What is your highest level of education?"] = "No Answer"
#plot_discrete_feature(df, 'What is your highest level of education?')

# education grouped
df["Did you go to college?"] = "No or Unknown"
df.loc[df['What is your highest level of education?']=='Some college',"Did you go to college?"] = "Yes"
df.loc[df['What is your highest level of education?']=='College degree',"Did you go to college?"] = "Yes"
df.loc[df['What is your highest level of education?']=='Graduate degree',"Did you go to college?"] = "Yes"
p4 = plot_discrete_feature(df, 'Did you go to college?')
tab4 = Panel(child=p4, title='Education')

# race*** weird!!
df.loc[df['What is your race?']=='DK/REF',"What is your race?"] = "No Answer"
p5 = plot_discrete_feature(df, 'What is your race?')
tab5 = Panel(child=p5, title='Race')

"""


# questions and answers


# approve Donald Trump?
df.loc[df['Do you approve or disapprove of how Donald Trump is handling his job as president?']=='DK/REF',"Do you approve or disapprove of how Donald Trump is handling his job as president?"] = "No Answer"
p6 = plot_discrete_feature(df, 'Do you approve or disapprove of how Donald Trump is handling his job as president?')
tab6 = Panel(child=p6, title='Q1')

# love america? ** people who say NO less likely to agree w/ WN
df.loc[df['Would you say that you love America?']=='DK/REF',"Would you say that you love America?"] = "No Answer"
p7 = plot_discrete_feature(df, 'Would you say that you love America?')
tab7 = Panel(child=p7, title='Q2')

# Do you think that government policies should help those who are poor and struggling in America?
df.loc[df['Do you think that government policies should help those who are poor and struggling in America?']=='DK/REF',"Do you think that government policies should help those who are poor and struggling in America?"] = "No Answer"
p8 = plot_discrete_feature(df, 'Do you think that government policies should help those who are poor and struggling in America?')
tab8 = Panel(child=p8, title='Q3')

# Do you think that most white people in America are racist?**
df.loc[df['Do you think that most white people in America are racist?']=='DK/REF',"Do you think that most white people in America are racist?"] = "No Answer"
p9 = plot_discrete_feature(df, 'Do you think that most white people in America are racist?')
tab9 = Panel(child=p9, title='Q4')

# Have you lost any friendships or other relationships as a result of the 2016 presidential election?
df.loc[df['Have you lost any friendships or other relationships as a result of the 2016 presidential election?']=='DK/REF',"Have you lost any friendships or other relationships as a result of the 2016 presidential election?"] = "No Answer"
p10 = plot_discrete_feature(df, 'Have you lost any friendships or other relationships as a result of the 2016 presidential election?')
tab10 = Panel(child=p10, title='Q5')

# *** Do you think it is likely or unlikely that there will be a Civil War in the United States within the next decade?
df.loc[df['Do you think it is likely or unlikely that there will be a Civil War in the United States within the next decade?']=='DK/REF',"Do you think it is likely or unlikely that there will be a Civil War in the United States within the next decade?"] = "No Answer"
p11 = plot_discrete_feature(df, 'Do you think it is likely or unlikely that there will be a Civil War in the United States within the next decade?')
tab11 = Panel(child=p10, title='Q6')

# Have you ever gone hunting?
df.loc[df['Have you ever gone hunting?']=='DK/REF',"Have you ever gone hunting?"] = "No Answer"
p12 = plot_discrete_feature(df, 'Have you ever gone hunting?')
tab12 = Panel(child=p12, title='Q7')

# Have you ever eaten a kale salad?
df.loc[df['Have you ever eaten a kale salad?']=='DK/REF',"Have you ever eaten a kale salad?"] = "No Answer"
p13 = plot_discrete_feature(df, 'Have you ever eaten a kale salad?')
tab13 = Panel(child=p13, title='Q8')

# *** super odd because he is black...
# If Dwayne "The Rock" Johnson ran for president as a candidate for your political party, would you vote for him?
df.loc[df['If Dwayne "The Rock" Johnson ran for president as a candidate for your political party, would you vote for him?']=='DK/REF','If Dwayne "The Rock" Johnson ran for president as a candidate for your political party, would you vote for him?'] = "No Answer"
q9 = plot_discrete_feature(df, 'If Dwayne "The Rock" Johnson ran for president as a candidate for your political party, would you vote for him?')
tab14 = Panel(child=p14, title='Q9')

# Who would you prefer as president of the United States, Darth Vader or Donald Trump?
df.loc[df['Who would you prefer as president of the United States, Darth Vader or Donald Trump?']=='DK/REF',"Who would you prefer as president of the United States, Darth Vader or Donald Trump?"] = "No Answer"
p15 = plot_discrete_feature(df, 'Who would you prefer as president of the United States, Darth Vader or Donald Trump?')
tab15 = Panel(child=p15, title='Q10')



# show as tabs - demographics
#layout = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])
#output_file('tabs.html')
#show(layout)

# show as tabs - questions
layout = Tabs(tabs=[tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15])
output_file('tabs.html')
show(layout)