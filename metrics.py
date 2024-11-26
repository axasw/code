from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(1)
warnings.filterwarnings("ignore")


class TrainAndValidate:
    def __init__(self, clf, X_encoded, y_label, test_id):
        self.X_encoded = X_encoded
        self.y_label = y_label

        self.test_id = test_id
        # remove 21 test data
        self.train_X_encoded = np.delete(X_encoded, self.test_id, axis=0)
        self.train_y_label = np.delete(y_label, self.test_id, axis=0)
        # shuffle the train data
        index = np.random.permutation(np.arange(len(self.train_y_label)))
        self.train_X_encoded = self.train_X_encoded[index]
        self.train_y_label = self.train_y_label[index]
        # split the test data
        self.test_X_encoded = self.X_encoded[self.test_id]
        self.test_y_label = self.y_label[self.test_id]

        self.classifier = clf

    def validate(self):

        self.classifier.fit(self.train_X_encoded, self.train_y_label)
        predicts = self.classifier.predict(X=self.test_X_encoded)

        # calculate metrics
        accuracy = accuracy_score(self.test_y_label, predicts)
        precision = precision_score(self.test_y_label, predicts, average='macro')
        recall = recall_score(self.test_y_label, predicts, average='macro')
        f1 = f1_score(self.test_y_label, predicts, average='macro')

        return {
            'acc': accuracy,
            'pre': precision,
            'recall': recall,
            'f1': f1
        }

    def get_weights(self):
        weights = self.classifier.coef_
        return weights


#######################################################

data = pd.read_excel(r'data.xlsx', header=1)
index = list(data[~data['conductivity S/m'].isna()].index)
data = data.iloc[index]
data = data.sort_values(by='conductivity S/m')

attrs = ['ligands', 'ions', 'addition ratio', 'reaction duration/h', 'solvent']
target = ['conductivity S/m']

X = data[attrs]
y = data[target]
X_encoded = pd.get_dummies(X, columns=attrs)
y = np.asarray(y).flatten()

ranks = [1.21042931324091E-06, 0.0147100137185219]

y_label = np.digitize(y, bins=ranks)
X_encoded = np.asarray(X_encoded) + 0

class1 = [0, 11, 33, 34, 65, 100, 111]
class2 = [124, 127, 128, 129, 130, 132, 133]
class3 = [140, 180, 188, 198, 201, 204, 215]

test_id = class1 + class2 + class3

models = [SVC(class_weight={0: 2, 1: 4.5, 2: 1.5}),
          RandomForestClassifier(),
          XGBClassifier(),
          KNeighborsClassifier()]

models_names = ['svm', 'random forest', 'xgboost', 'kNN']

for model, name in zip(models, models_names):
    tv = TrainAndValidate(model, X_encoded, y_label, class1 + class2 + class3)

    info = tv.validate()
    print(f'============={name}==============')
    print(f"acc={info['acc']}")
    print(f"pre={info['pre']}")
    print(f"recall={info['recall']}")
    print(f"f1={info['f1']}")

##################### plot ############################
# data = {
#     'Model': ['RF', 'SVM', 'XGBoost', 'k-NN'] * 4,
#     'Metric': ['Accuracy'] * 4 + ['Precision'] * 4 + ['Recall'] * 4 + ['F1 score'] * 4,
#     'Value': [0.5238095238095238, 0.8571428571428571, 0.7142857142857143, 0.5714285714285714,
#               0.6805555555555557, 0.8611111111111112, 0.7142857142857143, 0.6931818181818182,
#               0.5238095238095238, 0.8571428571428572, 0.7142857142857143, 0.5714285714285715,
#               0.4809941520467836, 0.8564102564102564, 0.7142857142857143, 0.5555555555555555]
# }
# df = pd.DataFrame(data)
#
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14
#
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Model', y='Value', hue='Metric', data=df, palette='muted', ci='sd', width=0.6)
#
# plt.xlabel('Model', fontdict={'fontsize': 18})
# plt.ylabel('Value', fontdict={'fontsize': 18})
# plt.legend(title='Metric', loc='upper right', ncol=2)
# plt.tight_layout()
# plt.savefig('./figs/metrics.png', bbox_inches='tight')
# plt.show()
#################################################