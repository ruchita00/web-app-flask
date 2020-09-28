import csv



import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt


import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier

import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

with open('D:\\indian_liver_patient.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
df=pd.read_csv('D:\\indian_liver_patient.csv')
for row in df:
        print(row)

df.shape
df.columns
df.head()
df.dtypes[df.dtypes=='object']
# Plot histogram grid
df.hist(figsize=(15,15), xrot=-45, bins=10) ## Display the labels rotated by 45 degress
plt.show()
df.describe()
## if score==negative, mark 0 ;else 1
def partition(x):
    if x == 2:
        return 0
    return 1

df['Dataset'] = df['Dataset'].map(partition)
df.describe(include=['object'])
df[df['Gender'] == 'Male'][['Dataset', 'Gender']].head()
sns.factorplot (x="Age", y="Gender", hue="Dataset", data=df);
model = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=4)
