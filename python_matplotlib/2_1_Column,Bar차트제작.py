import pandas as pd

marathon_2015_2017 = pd.read_csv("D:\creapple\python_matplotlib\data/marathon_2015_2017.csv")

import matplotlib.pyplot as plt
import seaborn as sns

print(marathon_2015_2017.head())

USA_runner = marathon_2015_2017[marathon_2015_2017.Country == 'USA']
print(USA_runner.info())


# 주별 사람수 
# plt.figure(figsize=(30,20))
# runner_state = sns.countplot('State', data = USA_runner)
# runner_state.set_title('Number of Runner by State - USA', fontsize = 18)
# runner_state.set_xlabel('State', fontdict = {'size': 16})
# runner_state.set_ylabel('Number of Runner', fontdict = {'size':16})
# plt.show()


#주별 사람수(남녀구별)
# plt.figure(figsize=(30,20))
# runner_state = sns.countplot('State', data = USA_runner, hue="M/F", palette = {'F' : 'r', "M" : 'b'}) #palette는 bar색깔
# runner_state.set_title('Number of Runner by State - USA', fontsize = 18)
# runner_state.set_xlabel('State', fontdict = {'size': 16})
# runner_state.set_ylabel('Number of Runner', fontdict = {'size':16})
# plt.show()


#주별 사람수(년도)
plt.figure(figsize=(30,20))
runner_state = sns.countplot('State', data = USA_runner, hue="Year") #palette는 bar색깔
runner_state.set_title('Number of Runner by State - USA', fontsize = 18)
runner_state.set_xlabel('State', fontdict = {'size': 16})
runner_state.set_ylabel('Number of Runner', fontdict = {'size':16})
plt.show()