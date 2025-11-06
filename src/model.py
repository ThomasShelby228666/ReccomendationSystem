import pandas as pd
import pickle

class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame, sim_matrix_path: str) -> None:
        self.df = df
        self.indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
        with open(sim_matrix_path, "rb") as f:
            self.sim_matrix = pickle.load(f)

    def get_recommendations(self, title: str, top_n: int=5) -> DataFrame:
        idx = self.indices[title]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : top_n + 1]
        movie_indices = [i[0] for i in sim_scores]
        return self.df.loc[movie_indices][["title", "vote_average", "overview"]]