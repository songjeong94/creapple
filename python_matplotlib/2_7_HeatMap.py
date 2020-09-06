
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

marathon_2015_2017 = pd.read_csv("D:\creapple\python_matplotlib\data/marathon_2015_2017.csv")

sns.set()

# Creatr marathon_2015_2017_under60 dataframe under age 60
marathon_2015_2017_under60 = marathon_2015_2017[marathon_2015_2017.Age.isin(range(0,60))]
# Counting by age, Male and Female 
marathon = marathon_2015_2017_under60.groupby('Age')['M/F'].value_counts().unstack().fillna(0)
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(10, 20))
sns.heatmap(marathon, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.show()


