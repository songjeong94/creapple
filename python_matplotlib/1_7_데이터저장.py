# 3개의 csv파일을 합치기

import pandas as pd

marathon_2015 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2015.csv")
marathon_2016 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2016.csv")
marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")

marathon_2015_2017 = pd.concat([marathon_2015, marathon_2016, marathon_2017], ignore_index=True, sort=False).set_index('Official Time')

print(marathon_2015_2017.describe())
#describe를 통해서 통계자료를 살펴볼수 있다.

print(marathon_2015_2017.sort_values(by=['Age']))
# 나이순으로 정렬해서 보기

print(marathon_2015_2017.sort_values(by='Age', ascending=False))
# 내림 차순

marathon_2015_2017.to_csv("D:\creapple\python_matplib\data/marathon_2015_2017.csv", index = None, header=True)
# 새로운 csv 파일로 저장
