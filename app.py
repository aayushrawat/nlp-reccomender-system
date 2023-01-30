import pickle
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.header('Movie Recommender System')
movies = pickle.load(open('movies.pkl', 'rb'))

cv = CountVectorizer(max_features = 5000, stop_words = "english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

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





