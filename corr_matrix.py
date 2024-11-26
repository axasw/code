import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel(r'data.xlsx', header=1)
index = list(data[~data['conductivity S/m'].isna()].index)
data = data.iloc[index]
data = data.sort_values(by='conductivity S/m')

attrs = ['ligands', 'ions', 'addition ratio', 'reaction duration/h', 'solvent']
target = ['conductivity S/m']

X = data[attrs]
y = data[target]

for attr in attrs:
    X[attr] = X[attr].factorize()[0]

X.columns = ['ligands', 'ions', 'addition ratio','reaction duration', 'solvent']

correlation_matrix = X.corr()
correlation_matrix = np.abs(correlation_matrix)
print(np.asarray(correlation_matrix))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True,
                      cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('./figs/corr_matrix.png', bbox_inches='tight')

plt.show()