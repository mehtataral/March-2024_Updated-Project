# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:24:56 2023

@author: Livewire
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 =  pd.read_csv(r"E:\Taral\DataScience\Project\msdData.csv",dayfirst =True, encoding='latin1',parse_dates=["Date"])
# df1 = pd.read_excel(r"E:\Taral\DataScience\Project\msdData.xlsx",parse_dates= ['Date'])
# print(df1)
# import warnings

# warnings.filterwarnings("ignore")
# df1['Date'] = pd.to_datetime(df1['Date'], errors='coerce')
# df1 = pd.to_datetime(df1.Date,format="%d/%m/%y")

df1.isnull().sum()
df1.dropna(inplace =True)
df1.isnull().sum()
df1



df1.Versus.unique()
df1.info()


df1.drop(columns = ["Sr. No."],inplace =True)
df1["Month"]= df1.Date.dt.month_name()
df1["Month"]
df1["Year"]= df1.Date.dt.year
df1["Year"]

month  =  df1.Month.unique()
month

month_data ={}
for x in month:
    filtered_df  = df1.loc[df1.Month == x]

    print(f"Month {x}:")
    print(filtered_df)
    
    month_data[x]=filtered_df

Jan = month_data["January"]
Jan.info()

Jan
Jan["Runs"]= Jan.Runs.astype(int) 
Jan["Date"] = Jan["Date"].astype(str)
Jan.Runs.unique()
Jan["Runs"]=Jan["Runs"].str.replace("-","0")
Jan.Runs.unique()
Jan["Runs"]= Jan.Runs.astype(int) 
jan_overall= Jan.groupby("Versus").sum()
# jan_overall= Jan.groupby("Versus").count()
jan_overall.reset_index(inplace =True)
jan_overall.drop(columns = ["Aggr","Avg","S/R.1","Year"],inplace =True)
jan_overall

Feb = month_data["February"]
Feb.info()

Feb
Feb["Runs"]= Feb.Runs.astype(int) 
Feb["Date"] = Feb["Date"].astype(str)
Feb.Runs.unique()
Feb["Runs"]=Feb["Runs"].str.replace("-","0")
Feb.Runs.unique()
Feb["Runs"]= Feb.Runs.astype(int) 

feb_overall= Feb.groupby("Versus").sum()
feb_overall.reset_index(inplace =True)
feb_overall
feb_overall.drop(columns = ["Aggr","Avg","S/R.1","Year"],inplace =True)
feb_overall


# Mar["Runs"].dtype

def operation(month_arg):
    x_month = month_data[month_arg]
    x_month.info()
    print("Month Name",month_arg)
    # print("before if ")
    if x_month["Runs"].dtype == 'int32':
        print("IN if ")
        x_month["Runs"]= x_month.Runs.astype(int)
        x_month.Runs.unique()
    else:
        print("IN else ")
        x_month["Runs"]=x_month["Runs"].str.replace("-","0")
        x_month.Runs.unique()
        x_month["Runs"]= x_month.Runs.astype(int) 
    x_month_overall= x_month.groupby("Versus").sum()
    x_month_overall.reset_index(inplace =True)
    print("Month Name",x_month_overall) 
    x_month_overall.drop(columns = ["Aggr","Avg","S/R.1","Year"],inplace =True)
    print("Month Name",x_month_overall) 
    x_month_overall = x_month_overall.sort_values("Runs",ascending =False).iloc[:]
   
    plt.figure(figsize=(12,8))
    sns.barplot(data = x_month_overall,x = "Versus",y = "Runs")
    plt.title("Analysis of {} ".format(month_arg))
    
operation("January")
operation("February") 
operation("March")
operation("April")
operation("May")
operation("June")
operation("July")
operation("August")
operation("September")
operation("October")
operation("November")
operation("December")


Mar = month_data["March"]
Apr = month_data["April"]
May = month_data["May"]
June = month_data["June"]
Jul = month_data["July"]
Aug = month_data["August"]
Sep = month_data["September"]
Oct = month_data["October"]
Nov = month_data["November"]
Dec = month_data["December"]
    
df1.info()
df1.Year = df1.Year.astype(str)
df1.Year
df1.Year.unique()

df1.Runs = df1.Runs.astype(int)
df1.Runs =df1.Runs.str.replace("-","0")
df1.Runs = df1.Runs.astype(int)

year_runs  = df1.groupby("Year").sum().sort_values("Runs",ascending =False)
year_runs.reset_index(inplace =True)
year_runs.drop(columns = ["Aggr","Avg","S/R.1"],inplace =True)
year_runs



plt.figure(figsize=(12,8))
sns.barplot(data = year_runs,x = "Year",y = "Runs")

# -----------------------------------------------------------
df1.info()
df1.Ground.unique()
df1

ground = df1.groupby("Ground").sum().sort_values("Runs",ascending =False).iloc[:5]
ground.reset_index(inplace =True)
ground


plt.figure(figsize=(12,8))
sns.barplot(data = ground,x = "Ground",y = "Runs")

# import requests
# import pandas as pd

# # Example API endpoint (hypothetical)
# api_url = "https://example-cricket-stats-api.com/player/ms-dhoni"

# # Make an API request to get Dhoni's statistics
# response = requests.get(api_url)

# if response.status_code == 200:
#     data = response.json()
#     runs_data = data['runs_data']  # Hypothetical data structure

#     # Convert data to a DataFrame for easier analysis
#     df = pd.DataFrame(runs_data, columns=['Ground', 'Runs'])

#     # Find the ground with the most runs
#     most_runs_ground = df[df['Runs'] == df['Runs'].max()]

#     print("Ground where Dhoni has scored the most runs:")
#     print(most_runs_ground)
# else:
#     print("Failed to retrieve data")

# The above code is a simplified example and does not access real data.

#--------------------------------------------------------------------

# 🔍 Problem 1: Analysis of Virat's top 5 ground performance in the year 2021?

# 🔍 Problem 2: Analyze Virat batting performance data and show great performance monthly and quarterly in the year 2018?

# 🔍 Problem 3: In which opposition team did Virat have the most top 5 batting performances in the year 2021?

# 🔍 Problem 4 : How many cricket matches did Virat play for India in the year 2017, 2018, 2019 & 2020.
