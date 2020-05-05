import pandas as pd
import numpy as np

# reading csv files
raw_movies = pd.read_csv('movies.csv')
raw_rate = pd.read_csv('ratings.csv')

# indexing movies by id of films
movies = raw_movies.set_index('movieId')

votes = raw_rate['movieId'].value_counts()
movies_rate = raw_rate.groupby('movieId').mean()['rating']

# adding new columns to movies
movies['total_votes'] = votes
movies['rate'] = movies_rate

# organizing movies by parameters
movies_by_votes = movies.sort_values('total_votes', ascending=False)
movies_by_rate = movies.sort_values('rate', ascending=False)

# RECOMMENDATION FOR NEW USERS: model using rate based in votes
model_for_new_users = movies_by_votes.query('rate > 4.0')

total_users = raw_rate['userId'].unique()


#  ---------------
def vetorial_distance(a, b):
    value = np.linalg.norm(a - b)
    return value


def user_average(user_rates, n=None):
    raw_notes_user = raw_rate.query('userId ==  %d' % user_rates)
    user_notes = raw_notes_user[['movieId', 'rating']].set_index('movieId')
    if n:
        user_notes = user_notes[:n]
    return user_notes


def distance_users(user_a, user_b, min_=5):
    a = user_average(user_a)
    b = user_average(user_b)

    named_a = '_%d' % user_a
    named_b = '_%d' % user_b

    named_a_str = 'rating{}'.format(named_a)
    named_b_str = 'rating{}'.format(named_b)

    compared_notes = a.join(b, lsuffix=named_a, rsuffix=named_b).fillna(0)

    if len(compared_notes) < min_:
        return None

    distance = vetorial_distance(compared_notes[named_a_str], compared_notes[named_b_str])

    return [user_a, user_b, distance]


def distance_from_all(user_x, users_to_analysis=None, users=total_users):
    if users_to_analysis:
        users = total_users[:users_to_analysis]
    distances = [distance_users(user_x, user_id) for user_id in users]
    distances = list(filter(None, distances))
    distances = pd.DataFrame(distances, columns=['user', 'other_user', 'distance'])
    return distances

def close_to(id_, closest = 300, users_to_analysis = None):
    distances = distance_from_all(id_, users_to_analysis = users_to_analysis)
    distances = distances.sort_values('distance')
    distances = distances.set_index('other_user').drop(id_)
    return distances.head(closest)


def movies_to(user):
    similars = close_to(user)
    similar_users = similars.index

    similar_notes = raw_rate.set_index('userId').loc[similar_users]
    similar_notes = similar_notes.groupby('movieId').mean()
    recommendation = similar_notes.sort_values('rating', ascending=False).join(movies)
    recommendation = recommendation.query('total_votes > 100')
    return recommendation