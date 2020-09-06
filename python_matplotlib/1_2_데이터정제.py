import pandas as pd

marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")

print(marathon_2017.isnull().sum(axis=0))
#  각 칼람에 null 값이 얼마나 있는지 확인한다.

print(marathon_2017.columns)
#각 칼럼을 확인

marathon_2017_clean = marathon_2017.drop(['Unnamed: 0', 'Bib', 'Unnamed: 9'], axis = 'columns')
# 특정한 컬럼을 없애준다.

print(marathon_2017_clean.head())

