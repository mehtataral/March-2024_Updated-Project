# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:21:47 2024

@author: Administrator
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"C:\Users\Administrator\Downloads\archive (5)\matches.csv")
df1
df1.info()

df1.winner
top4_winner=df1.groupby("winner").count().reset_index()
top4_winner =top4_winner.sort_values("team1",ascending = False).iloc[:4,:]
top4_winner


plt.figure(figsize=(12,8))
sns.barplot(data = top4_winner,x = "winner",y= "team1")

#------------------------------------------------------------------

bot4_winner=df1.groupby("winner").count().reset_index()
bot4_winner =bot4_winner.sort_values("team1",ascending = True).iloc[:4,:]
bot4_winner


plt.figure(figsize=(12,8))
sns.barplot(data = bot4_winner,x = "winner",y= "team1")

#------------------------------------------------------------------

India = df1.loc[df1.winner == "India"]
India.info()


India_data = df1.loc[(df1.team1 == "India")|(df1.team2 == "India")]
India_data
India_data.info()


player_of_match =  India_data.groupby("player_of_match").count().reset_index().sort_values("team1",ascending =False)
player_of_match


plt.figure(figsize=(12,8))
sns.barplot(data = player_of_match,x = "player_of_match",y = "team1")
#------------------------------------------------------------------
# I want to calculate wining percentage after winning toss and match

india_won_toss = India_data[India_data.toss_winner =="India"]
india_won_toss

India_win_percenatge  = (india_won_toss.winner =="India").mean() *100
India_win_percenatge

# check_freq of taking bat or bowl

India_bat_bowl  = india_won_toss.toss_decision.value_counts()
India_bat_bowl

plt.pie(India_bat_bowl,labels =["bat","Field"],autopct= "%1.0f%%")
plt.legend()
bat_win_rate = (india_won_toss[india_won_toss['toss_decision'] == 'bat']['winner'] == 'India').mean() * 100
bat_win_rate
bowl_win_rate = (india_won_toss[india_won_toss['toss_decision'] == 'field']['winner'] == 'India').mean() * 100
bowl_win_rate
(india_won_toss.toss_decision == "bat").mean()*100
(india_won_toss.toss_decision == "field").mean()*100

# --------------------------------------------------------------

India_loss_toss = India_data[India_data.toss_winner != "India"]
India_loss_toss

India_loss_perc = (India_loss_toss.winner =="India").mean()*100
India_loss_perc

India_loss_bat_bowl = India_loss_toss.toss_decision.value_counts()
India_loss_bat_bowl

plt.pie(India_loss_bat_bowl,labels =["bat","Field"],autopct= "%1.0f%%")
plt.legend()

(India_loss_toss["toss_decision"] =="field").mean()*100
(India_loss_toss["toss_decision"] =="bat").mean()*100


bat_win = (India_loss_toss[India_loss_toss['toss_decision'] == 'bat']['winner'] == 'India').mean() * 100
bat_win

bowl_win = (India_loss_toss[India_loss_toss['toss_decision'] == 'field']['winner'] == 'India').mean() * 100
bowl_win

# --------------------------------------------------------------

not_get_bat = (India_loss_toss.loc[India_loss_toss["toss_decision"]=="bat"]["winner"]=="India").mean()*100
get_bat = (India_loss_toss.loc[India_loss_toss["toss_decision"] =="field"]["winner"] =="India").mean()*100


# ------------------------------------------------------------------
India
India.info()

India.venue
India.city






































