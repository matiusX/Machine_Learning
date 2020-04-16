import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


SEED = 5
np.random.seed(SEED)

model = SVC()

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw' \
      '/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv '
db = pd.read_csv(uri)


def transform_data(data):
    # rename columns
    new_names = {
        'mileage_per_year': 'mi_y',
        'price': 'value'
    }
    data = data.rename(columns=new_names)

    # change yes and no for 1 and 0
    change_for_binary = {
        'yes': 1,
        'no': 0
    }
    data['sold'] = data.sold.map(change_for_binary)

    # giving an age to car
    year = datetime.today().year
    data['age'] = year - data.model_year

    # mi to km
    k = 1.60934
    data['km_y'] = data.mi_y * k

    # removing unused columns
    data = data.drop(columns=["Unnamed: 0", "mi_y", "model_year"], axis=1)

    # modeling data
    data = data[['km_y', 'value', 'age', 'sold']]

    return data


db = transform_data(db)

x = db[['km_y', 'value', 'age']]
y = db['sold']

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y,
                                                            stratify=y,
                                                            train_size=0.75)

to_train, to_test = len(raw_train_x), len(raw_test_x)
print('To train: {}. To test: {}'.format(to_train, to_test))

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model.fit(train_x, train_y)
preview = model.predict(test_x)
win_rate = accuracy_score(test_y, preview) * 100
print('Win-rate: {:.2f}%'.format(win_rate))
