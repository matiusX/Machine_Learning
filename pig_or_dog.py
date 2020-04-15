from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
# features(yes[1], no[0])
# [long fur, short leg, say au-au]

pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

# dog->0 and pig->1
train_x = [pig1, pig2, pig3, dog1, dog2, dog3]
train_y = [1, 1, 1, 0, 0, 0]
model.fit(train_x, train_y)

animal_a = [1, 0, 1]
animal_b = [1, 1, 0]
animal_c = [0, 1, 1]


testes_x = [animal_a, animal_b, animal_c]
testes_y = [0, 1, 1]

preview = model.predict(testes_x)

win_rate = accuracy_score(testes_y, preview)

print('The win-rate is: {}'.format(win_rate * 100))



