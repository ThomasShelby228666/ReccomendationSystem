import streamlit as st
import pandas as pd
from src.model import ContentBasedRecommender
from src.recommend import get_movie_recommendations

#  streamlit run c:\users\user\desktop\recommendationsystem\app.py

@st.cache(allow_output_mutation=True)
def load_recommender() -> ContentBasedRecommender:
    df = pd.read_csv("data/TMDB_movie_dataset_v11.csv")
    df = df.sort_values("popularity", ascending=False).head(125000).reset_index(drop=True)
    df = df.dropna(subset=["overview"])

    # threshold = df["vote_count"].quantile(0.7)
    # df = df[df["vote_count"] > threshold].copy()

    recommender = ContentBasedRecommender(df, 'models/similarity_matrix.pkl')
    return recommender

recommender = load_recommender()

# def get_recommendations(title: str, top_n: int=5):
#     idx = indices[title]
#     sim_scores = list(enumerate(sim_matrix[idx].toarray().flatten()))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1 : top_n + 1]
#     movie_indices = [i[0] for i in sim_scores]
#     return df.iloc[movie_indices][["title", "vote_average", "overview"]]

st.title("Рекомендательная система фильмов")
title = st.text_input("Введите название фильма")

# if st.button("Получить рекомендации"):
#     if title not in indices:
#         st.error(f"Фильм '{title}' не найден в датасете.")
#     else:
#         try:
#             recs = get_recommendations(title, top_n=5)
#             st.write("Рекомендации:")
#             st.dataframe(recs[["title", "vote_average", "overview"]])
#         except Exception as e:
#             st.error(f"Ошибка: {e}")

if st.button("Получить рекомендации"):
    try:
        recs = get_movie_recommendations(title, recommender, top_n=5)
        st.write("Рекомендации: ")
        st.dataframe(recs[["title", "vote_average", "overview"]])
    except ValueError as e:
        st.error(str(e))