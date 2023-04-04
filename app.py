import pandas as pd
import numpy as np
import ast
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

cred = "https://raw.githubusercontent.com/aayushrawat/mrs-dataset-files/main/tmdb_5000_credits.csv"
mov = "https://raw.githubusercontent.com/aayushrawat/mrs-dataset-files/main/tmdb_5000_movies.csv"
movie = pd.read_csv(mov)
credit = pd.read_csv(cred)
movie = movie.merge(credit, on = "title")
movie = movie[["title", "genres", "movie_id", "overview", "cast", "crew", "keywords"]]


def normaltext(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i["name"])
    return l


movie["genres"] = movie["genres"].apply(normaltext)
movie["keywords"] = movie["keywords"].apply(normaltext)


def normaltext3(obj):
    l = []
    c = 0
    for i in ast.literal_eval(obj):
        if c != 3:
            l.append(i["name"])
            c += 1
        else:
            break
    return l


movie["cast"] = movie["cast"].apply(normaltext3)


def getdirector(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            l.append(i["name"])
            break
    return l


movie["crew"] = movie["crew"].apply(getdirector)
movie["genres"] = movie["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movie["keywords"] = movie["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movie["cast"] = movie["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movie["crew"] = movie["crew"].apply(lambda x: [i.replace(" ", "") for i in x])
backup_df = movie

x = list(movie["overview"])


def nikalo(o):
    if isinstance(o, float):
        print(o)
    else:
        pass


movie["overview"] = movie["overview"].fillna(" ")
movie["overview"] = movie["overview"].apply(lambda x: x.split())
movie["tags"] = movie["cast"] + movie["keywords"] + movie["overview"] + movie["crew"] + movie["genres"]
new = movie[["movie_id", "title", "tags"]]
new["tags"] = new["tags"].apply(lambda x: " ".join(x))
new["tags"] = new["tags"].apply(lambda x: x.lower())

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new["tags"] = new["tags"].apply(stem)
movies = new
cv = CountVectorizer(max_features = 5000, stop_words = "english")
vectors = cv.fit_transform(new["tags"]).toarray()
similarity = cosine_similarity(vectors)


st.header('Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown", movie_list)


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommender(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movielist = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendation = []
    for i in movielist:
        movie_id = movies.iloc[i[0]].movie_id
        poster = fetch_poster(movie_id)
        title = movies.iloc[i[0]].title
        recommendation.append((title, poster))
    return recommendation


if st.button('Show Recommendation'):
    recommendations = recommender(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommendations[0][0])
        st.image(recommendations[0][1])
    with col2:
        st.text(recommendations[1][0])
        st.image(recommendations[1][1])

    with col3:
        st.text(recommendations[2][0])
        st.image(recommendations[2][1])
    with col4:
        st.text(recommendations[3][0])
        st.image(recommendations[3][1])
    with col5:
        st.text(recommendations[4][0])
        st.image(recommendations[4][1])
