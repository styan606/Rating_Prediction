import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_excel(r"C:\Users\styli\OneDrive\Desktop\excel\PLEASE.xlsx", engine='openpyxl')

#df_imgs = df.iloc[:, -2:]
#df = df.iloc[:, :-2]

#df['RandomValue'] = np.random.rand(len(df))
data = df.values

# split into input and output elements
X, y = data[:, 1:], data[:, 0]
# label encode the target variable
y = LabelEncoder().fit_transform(y) + 1

counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#pyplot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.xlabel("Class")
pyplot.ylabel("Frequency")
pyplot.title("Class Distribution")
pyplot.show()

strat = {1: 1500, 2: 1100, 3: 1300, 4: 1400}

sm = SMOTENC(categorical_features=[0,2,3,4,5,9,10,11,12,13,14],sampling_strategy=strat, random_state=0)
X, y = sm.fit_resample(X, y)

# summarize distribution
counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
#pyplot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.xlabel("Class")
pyplot.ylabel("Frequency")
pyplot.title("Class Distribution")
pyplot.show()

# Convert 'X' into a DataFrame for easier manipulation
X_df = pd.DataFrame(X, columns=df.columns[1:])

# Convert 'y' to a DataFrame (since it's a 1D array)
y_df = pd.DataFrame(y, columns=['rating'])

# Concatenate 'y' with 'X'
ndata = pd.concat([y_df, X_df], axis=1)

#ndata.to_excel(r"C:\Users\Sendsteps\Desktop\SMOTENC.xlsx", index=False, engine='openpyxl')

# Compute the correlation matrix
corr_matrix = ndata.corr()
pyplot.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,            # Display correlation values on the heatmap
    cmap='coolwarm',       # Color map (can be 'viridis', 'coolwarm', etc.)
    center=0,              # Center the color bar at zero
    square=True,           # Make the cells square-shaped
    fmt=".2f",
    annot_kws={"size": 8} # Format for correlation values
)
pyplot.show()

'''
for i in range(760):
    n = response[i].split('@@')
    newString = ''
    for element in n:
        v = json.loads(element)
        #print(i, 'good')
        index = v['choices'][0]['message']['content'].replace("\n", " ")
        newString += index + ';'
    df.replace(response[i], newString, inplace=True)
'''
