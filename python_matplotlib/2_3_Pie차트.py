import pandas as pd

marathon_2015_2017 = pd.read_csv("D:\creapple\python_matplotlib\data/marathon_2015_2017.csv")

import matplotlib.pyplot as plt

labels = 'Male', 'Female'
#직접 레이블을 줬음 csv파일에는 m/f로 표기도이었음

explode = (0, 0.1)
#차트를 입체적으로 보기위함

plt.figure(figsize=(7,7))

plt.pie(marathon_2015_2017['M/F'].value_counts(), explode=explode, labels=labels, startangle=90, shadow=True, autopct='%.1f')
plt.title("Male vs. Female", fontsize=18)
plt.show()