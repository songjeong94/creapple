import pandas as pd

marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")
marathon_2017_clean = marathon_2017.drop(['Unnamed: 0', 'Bib', 'Unnamed: 9'], axis = 'columns')


marathon_2017_clean['Senior'] = marathon_2017_clean.Age > 60
#senior 컬럼을 만들어서 부울 형식의 60이상의 사람이 모인 컬림을 추가
print(marathon_2017_clean.head())

marathon_2017_clean['Year'] = '2017'
# 여러 csv를 합칠때 현재 칼럼데이터의 구분값을 넣어줌
print(marathon_2017_clean.head())


KEN_runner = marathon_2017_clean
