import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 50
np.random.seed(SEED)
model = SVC()

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw' \
      '/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv '

data = pd.read_csv(uri)
names = {
    'unfinished': 'not_done',
    'expected_hours': 'hours',
    'price': 'value',
}
data = data.rename(columns=names)
change = {
    0: 1,
    1: 0
}
data['done'] = data.not_done.map(change)
x = data[['hours', 'value']]
y = data['done']

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y,
                                                            test_size=0.25,
                                                            stratify=y,
                                                            )
print('To train: {}. To test: {}'.format(len(raw_train_x), len(raw_test_x)))

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model.fit(train_x, train_y)
preview = model.predict(test_x)

win_rate = accuracy_score(test_y, preview) * 100
print("Win-rate-real: %.2f%%" % win_rate)

sbn.scatterplot(x='hours', y='value', hue='done', data=data)
plt.show()
# :)
