import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
model = LinearSVC()
SEED = 20
uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw' \
      '/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv '
data = pd.read_csv(uri)
x = data[['home', 'how_it_works', 'contact']]
y = data['bought']

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=SEED,
                                                    stratify=y,
                                                    test_size=0.25)

print('To train, we use {} and to test, we use {}'.format(len(train_x), len(test_x)))

# .fit() to train the model
model.fit(train_x, train_y)
# .predict() to test, based on model.fit(), to give values for preview
preview = model.predict(test_x)
# accuracy_score() to compare preview w/ real values
win_rate = accuracy_score(test_y, preview) * 100

print("Win-rate: %.2f%%" % win_rate)
# :)
