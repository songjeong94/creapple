#산포도차트

import pandas as pd
import matplotlib.pyplot as plt

marathon_2015_2017 = pd.read_csv("D:\creapple\python_matplotlib\data/marathon_2015_2017.csv")


# Select male and female runners by conditional expression
MALE_runner = marathon_2015_2017[marathon_2015_2017['M/F'] == 'M']
FEMALE_runner = marathon_2015_2017[marathon_2015_2017['M/F'] == 'F']


plt.figure(figsize=(20,20))

x_male = MALE_runner.Age
y_male = MALE_runner['Official Time']

x_female = FEMALE_runner.Age
y_female = FEMALE_runner['Official Time']

plt.plot(x_male, y_male, '.', color='b', alpha=0.5)
plt.plot(x_female, y_female, '.', color='r', alpha=0.5)
# Generate labels and title
plt.xlabel("Age", fontsize=16)
plt.ylabel("Official Time (Second)",fontsize=16)
plt.title("Distribution by Running Time and Age",fontsize=20)
# Show plot
plt.show()

