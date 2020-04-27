import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# elbow_cluster = 15
kmeans = KMeans(n_clusters=3)
scaler = StandardScaler()

# accessing database
uri = 'https://raw.githubusercontent.com/alura-cursos/machine-learning-algoritmos-nao-supervisionados/master/movies.csv'
data = pd.read_csv(uri)


# rename columns in database
def personalize_films(database):
    new_names = {
        'movieId': 'id',
        'genres': 'gen'
    }
    database = database.rename(columns=new_names)
    return database


data = personalize_films(data)

# creation of binary genres
raw_genres = data.gen.str.get_dummies()
type_of_films = pd.concat([data, raw_genres], axis=1)
# scaling genres
genres = scaler.fit_transform(raw_genres)

kmeans.fit(genres)

aaa = pd.DataFrame(kmeans.cluster_centers_,
                         columns=raw_genres.columns).transpose()
clusters = []

max_iter = 15

for i in range(1, max_iter + 1):
    kmeans = KMeans(n_clusters=i, random_state=1234)
    kmeans.fit(genres)
    clusters.append((i,kmeans.inertia_,))
plt.plot([t[0] for t in clusters],[t[1] for t in clusters], marker="X")
