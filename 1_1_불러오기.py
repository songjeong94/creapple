import pandas as pd

marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")

print(marathon_2017.head())
# 앞에 5행

print(marathon_2017.info())
# 데이터의 구조를 가져온다.

