import pandas as pd

marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")

marathon_2017_clean = marathon_2017.drop(['Unnamed: 0', 'Bib', 'Unnamed: 9'], axis = 'columns')
# 특정한 컬럼을 없애준다.

names = marathon_2017_clean.Name
# name칼럼만 가져오기
print(names)

# official_time = marathon_2017_clean.Official Time
# 파이썬에서는 띄어쓰기가 인식되지 않는다
official_time = marathon_2017_clean["Official Time"]


print(official_time)
