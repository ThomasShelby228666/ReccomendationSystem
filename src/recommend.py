from .model import ContentBasedRecommender
import pandas as pd

def get_movie_recommendations(title: str, recommender: ContentBasedRecommender, top_n: int=5):
    if title not in recommender.indices:
        raise ValueError(f"Фильм '{title}' не найден в датасете")
    return recommender.get_recommendations(title, top_n=top_n)

def list_available_titles(recommender: ContentBasedRecommender, limit: int=10):
    return recommender.df["title"].head(limit).tolist()