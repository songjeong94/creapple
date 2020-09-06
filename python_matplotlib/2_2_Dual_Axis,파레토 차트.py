#
import pandas as pd
marathon_2015_2017 = pd.read_csv("D:\creapple\python_matplotlib\data/marathon_2015_2017.csv")

import matplotlib.pyplot as plt

runner_1860 = marathon_2015_2017[marathon_2015_2017.Age.isin(range(18,60))]
#Age칼럼에 18세_60세까지

runner_1860_counting = runner_1860['Age'].value_counts()

print(runner_1860.info())
print(runner_1860_counting)


x = runner_1860_counting.index
x = [str(i) for i in x]
#뽑아온 i 값은 스트링 타입으로 변경해줘야한다.
y = runner_1860_counting.values
ratio = y / y.sum()
#전체중의 각각의 비율
ratio_sum = ratio.cumsum()
#누적 퍼센티지

fig, barChart = plt.subplots(figsize=(20, 10))
barChart.bar(x, y)
lineChart = barChart.twinx()
#twinx는 x축을 공유
lineChart.plot(x, ratio_sum, '-ro', alpha=0.5)
#a= 투명도

ranges = lineChart.get_yticks()
lineChart.set_yticklabels(['{:,.1%}'.format(x) for x in ranges])
ratio_sum_percentages = ['{0: .0%}'.format(x) for x in ratio_sum]
for i, txt in enumerate(ratio_sum_percentages):
    lineChart.annotate(txt, (x[i], ratio_sum[i]), fontsize=14)

barChart.set_xlabel('Age', fontdict = {'size':16})
barChart.set_ylabel('Number of runner', fontdict={'size':16})
plt.title('Pareto Chart - Number of runner by Age', fontsize=18)
plt.show()
