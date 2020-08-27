#시간으로 되있는값을 초로 바꾸거나 훈련하기 편한 데이터형태로변환

import pandas as pd

marathon_2017 = pd.read_csv("D:\creapple\python_matplib\data/marathon_results_2017.csv")
marathon_2017_clean = marathon_2017.drop(['Unnamed: 0', 'Bib', 'Unnamed: 9'], axis = 'columns')

#시간을 초로 바꾼다(유저 함수로 구현)
def to_seconds(record):
    hms = record.str.split(':', n = 2, expand = True)
    return hms[0].astype(int) * 3600 + hms[1].astype(int) * 60 + hms[2].astype(int)

#official time sec라는 시간을 초로 바꾼 칼럼을 만들어준다
marathon_2017['Official Time Sec'] = to_seconds(marathon_2017['Official Time'])
print(marathon_2017.head())

#내장함수로 구현
import numpy as np
marathon_2017['Official Time Sec'] = pd.to_timedelta(marathon_2017['Official Time'])

marathon_2017['Official Time New'] = marathon_2017['Official Time Sec'].astype('m8[s]').astype(np.int64)
print(marathon_2017.head())