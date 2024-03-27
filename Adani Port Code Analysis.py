# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:50:34 2024

@author: Administrator
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df1 = pd.read_csv(r"C:\Users\Administrator\Downloads\ADANIPOWER.NS (1).csv")
df1
df1.info()

# 1. Highest closing price during the given period
Highest_closing_price  = df1.Close.max()
print("Highest Closing Price till now ",Highest_closing_price)



# 2. Date with the highest trading volume
df1.Volume.max()

df1["Volume"].idxmax()
df1.describe()
df1.loc[3115]

date_highest_volume = df1.loc[df1["Volume"].idxmax(), "Date"]
print("Date with the highest trading volume:", date_highest_volume)


# 3. Largest intraday price range
df1["Intraday Range"]  = df1.High  -  df1.Low
df1["Intraday Range"]
df1["Intraday Range"].max()
df1["Intraday Range"].idxmax()


largest_intraday_range_value = df1.loc[df1["Intraday Range"].idxmax()]
largest_intraday_range_value
print("Largest intraday price range:", largest_intraday_range_value)


# 4. Days where closing price exceeded opening price

close_price_exceed_open  = df1.loc[df1.Close > df1.Open]
close_price_exceed_open
print("close_price_exceed_open in days",len(close_price_exceed_open))

# 5. Average closing price
avg_price = df1.Close.mean()
avg_price

print("avg price of adani power till now ",avg_price)

# 6. Percentage change in closing price from March 27, 2023, to April 10, 2023

date_27_march = df1.loc[df1.Date == "2023-03-27","Close"]
date_27_march

date_10_april = df1.loc[df1.Date == "2023-04-10","Close"]
date_10_april

# diff_date_range = df1.loc[(df1.Date >= "2023-03-27")&(df1.Date <= "2023-04-10")]
# diff_date_range
# diff_date_range.Close

change_in_closing = (date_10_april - date_27_march)/(date_27_march) *100
change_in_closing

# 7. Date with the largest percentage increase in closing price compared to the previous day


# pct_change() function is used to calculate the percentage change between the current 
# and previous element in a Series or DataFrame. It's commonly used in time series analysis 
# and financial data analysis to calculate the percentage change between consecutive 
# observations.

df1["Percentage Change"]=df1.Close.pct_change()*100
df1["Percentage Change"]

df1["Percentage Change"].max()
df1["Percentage Change"].idxmax()
df1.loc[2259]
date_largest_percentage_increase = df1.loc[df1["Percentage Change"].idxmax()]
date_largest_percentage_increase

largest_percentage_increase = df1["Percentage Change"].max()
print("Date with the largest percentage increase in closing price:", date_largest_percentage_increase)
print("Largest percentage increase in closing price:", largest_percentage_increase)

# 8. Date(s) with the lowest closing price and corresponding value

df1.Close.min()
df1.Close.idxmin()

df1.loc[df1.Close.idxmin()]


# 9. Average daily trading volume
avg_vol = df1.Volume.mean()
avg_vol

# 10. Calculate the total rupees volume traded (total value of shares traded) for each day.
df1['Total_Rupees_Volume'] = df1['Close'] * df1['Volume']
print("Total rupees volume traded for each day:\n", df1[['Date', 'Total_Rupees_Volume']])

# 11. 50 days moving average & 200 daysmoving average
df1["50_MA"] = df1.Close.rolling(50).mean()
df1["50_MA"]

df1["200_MA"] = df1.Close.rolling(200).mean()
df1["200_MA"]

plt.figure(figsize=(10, 6))
plt.plot(df1['Date'], df1['Close'], label='Closing Price', color='blue')
plt.plot(df1['Date'], df1['50_MA'], label='50-Day MA', color='red', linestyle='--')
plt.plot(df1['Date'], df1['200_MA'], label='200-Day MA', color='green', linestyle='--')
plt.title('Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# 12. Identify bullish (golden) and bearish (death) crosses based on moving averages.
df1['Bullish_Cross'] = (df1['50_MA'] > df1['200_MA']) & (df1['50_MA'].shift(1) <= df1['200_MA'].shift(1))
df1['Bearish_Cross'] = (df1['50_MA'] < df1['200_MA']) & (df1['50_MA'].shift(1) >= df1['200_MA'].shift(1))
print("Bullish and bearish crosses:\n", df1[['Date', 'Bullish_Cross', 'Bearish_Cross']])

