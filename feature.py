from sklearn.svm import SVR, SVC
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import seaborn as sns

np.random.seed(1)
warnings.filterwarnings("ignore")

data = pd.read_excel(r'data.xlsx', header=1)
index = list(data[~data['conductivity S/m'].isna()].index)
data = data.iloc[index]
data = data.sort_values(by='conductivity S/m')

attrs = ['ligands', 'ions', 'addition ratio', 'reaction duration/h', 'solvent']
target = ['conductivity S/m']


X = data[attrs]
y = data[target]
y = np.asarray(y).flatten()

for attr in attrs:
    X[attr] = X[attr].factorize()[0]

model = SVR(kernel='linear')
model = model.fit(X, y)

w = model.coef_
w = np.abs(w.flatten())
w[0] = w[0] + 1e-2

plt.rcParams['font.family'] = 'Times New Roman'

data = {
    'rank': ['reaction duration', 'ions', 'addition ratio','ligands', 'solvent'],
    'value': list(w)
}
df = pd.DataFrame(data)
df = df.sort_values(by='value', ascending=False)

plt.figure(figsize=(6, 4))
ax = sns.barplot(x='value', y='rank', data=df, palette='viridis', orient='h', width=0.5)
print(df)

ax.set_xlabel('Importance', fontsize=18)
ax.set_ylabel('Feature', fontsize=18)


ax.tick_params(axis='both', which='major', labelsize=14)


plt.tight_layout()
plt.savefig('./figs/feature.png', bbox_inches='tight')
plt.show()
