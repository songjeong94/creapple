import pandas as pd

marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")

marathon_2017_clean = marathon_2017.drop(['Unnamed: 0', 'Bib', 'Unnamed: 9'], axis = 'columns')

print(marathon_2017_clean.info())

seniors = marathon_2017_clean.Age > 60
# 칼럼age가 60 이상인 사람을 확인
print(seniors)


KEN_runner = marathon_2017_clean[marathon_2017_clean.Country == 'KEN']
# country가 케냐인 사람을 가져오기
print(KEN_runner)
