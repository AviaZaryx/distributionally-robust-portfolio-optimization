import pandas as pd

data_bank = pd.read_csv('D:\Downloads\pythonProject1\\bank_stocks.csv')
data_10 = pd.read_csv('D:\Downloads\pythonProject1\combined_10_stocks.csv')
data_all = pd.read_csv('D:\Downloads\pythonProject1\combined_all_stocks.csv')
data_insurance = pd.read_csv('D:\Downloads\pythonProject1\insurance_stocks.csv')
data_other = pd.read_csv('D:\Downloads\pythonProject1\other_stocks.csv')
data_tech = pd.read_csv('D:\Downloads\pythonProject1\\tech_stocks.csv')
data_trade = pd.read_csv('D:\Downloads\pythonProject1\\trade_stocks.csv')

print(data_bank.head())
print(data_10.head())
print(data_all.head())
print(data_insurance.head())
print(data_other.head())
print(data_tech.head())
print(data_trade.head())