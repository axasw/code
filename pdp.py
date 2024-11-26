import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC
import pandas as pd
import warnings
from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay

np.random.seed(1)
warnings.filterwarnings("ignore")


data = pd.read_excel(r'data.xlsx', header=1)
index = list(data[~data['conductivity S/m'].isna()].index)
data = data.iloc[index]
data = data.sort_values(by='conductivity S/m')

attrs = ['ligands', 'ions', 'addition ratio', 'reaction duration/h', 'solvent']
target = ['conductivity S/m']

X = data[attrs]
y = np.asarray(data[target])
y = np.asarray(y).flatten()


for attr in attrs:
    temp1 = np.asarray(X[attr]).reshape(-1, 1)
    X[attr] = X[attr].factorize()[0]
    temp2 = np.asarray(X[attr]).reshape(-1, 1)
    temp3 = []
    for i, j in zip(temp1, temp2):
        temp3.append(f'{i} -> {j}')
    print(set(temp3))

model = SVR().fit(X, y)

features = [0, 1, 2, 3, 4]  # 选择特征的索引

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 1.5  # 设置线的粗细
plt.rcParams['lines.marker'] = 'o'

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
axs = axs.flatten()

results = PartialDependenceDisplay.from_estimator(
    model, X, features, feature_names=['ligands', 'ions', 'addition ratio', 'reaction duration', 'solvent'], ax=axs[:5],
    target=0)

axs[0].set_ylabel("Conductivity S/m", fontdict={'fontsize': 18})
axs[0].set_xlabel("ligands", fontdict={'fontsize': 18})
axs[0].text(0.9, 0.9, '(a)', transform=axs[0].transAxes, fontsize=18, va='center', ha='center')
axs[0].set_xticks([0, 1, 2, 3, 4])
axs[0].set_xticklabels(['-NH2', '-SH', '-OMe', '-OH', '-COOH'])
for line in axs[0].lines:
    y_data = line.get_ydata()
    print(y_data)

axs[1].set_ylabel("")
axs[1].set_xlabel("ions", fontdict={'fontsize': 18})
axs[1].text(0.9, 0.9, '(b)', transform=axs[1].transAxes, fontsize=18, va='center', ha='center')
axs[1].set_xticks([0, 1, 2])
axs[1].set_xticklabels([r"Ag(AgNO$_3$)", r"Fe(FeCl$_2$)", r"Cu(Cu$_2$O)"])
for line in axs[1].lines:
    y_data = line.get_ydata()
    print(y_data)

axs[2].set_ylabel("")
axs[2].set_xlabel("addition ratio", fontdict={'fontsize': 18})
axs[2].text(0.9, 0.9, '(c)', transform=axs[2].transAxes, fontsize=18, va='center', ha='center')
axs[2].set_xticks([0, 1, 2])
axs[2].set_xticklabels(['6:1', '1:1', '3:1'])
for line in axs[2].lines:
    y_data = line.get_ydata()
    print(y_data)

axs[3].set_ylabel("Conductivity S/m", fontdict={'fontsize': 18})
axs[3].set_xlabel("reaction duration", fontdict={'fontsize': 18})
axs[3].text(0.9, 0.9, '(d)', transform=axs[3].transAxes, fontsize=18, va='center', ha='center')
axs[3].set_xticks([0, 1, 2])
axs[3].set_xticklabels(['4', '12', '24'])
for line in axs[3].lines:
    y_data = line.get_ydata()
    print(y_data)

axs[4].set_ylabel("")
axs[4].set_xlabel("solvent", fontdict={'fontsize': 18})
axs[4].text(0.9, 0.9, '(e)', transform=axs[4].transAxes, fontsize=18, va='center', ha='center')
axs[4].set_xticks([0, 1])
axs[4].set_xticklabels(['ethanol', 'water'])
for line in axs[4].lines:
    y_data = line.get_ydata()
    print(y_data)

axs[5].set_visible(False)

plt.tight_layout()
plt.savefig('./figs/pdp.png', bbox_inches='tight')
plt.show()
