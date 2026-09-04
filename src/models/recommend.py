import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class ContentRecommender:
    def __init__(self):
        self.df = None
        self.tfidf = None
        self.similarity_matrix = None

    def load_data(self, data_path):
        """Load processed data"""
        self.df = pd.read_csv(data_path)
        return self

    def build_similarity_matrix(self):
        """Build similarity matrix"""
        if self.df is None:
            raise ValueError("Load data first")

        self.tfidf = TfidfVectorizer(max_features=5000)
        tfidf_matrix = self.tfidf.fit_transform(self.df["combined_features"])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        return self

    def get_recommendations(self, title, n=5):
        """Get recommendations for a title"""
        idx = self.df[self.df["title"] == title].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : n + 1]

        recommendations = []
        for i, score in sim_scores:
            recommendations.append(
                {
                    "title": self.df.iloc[i]["title"],
                    "type": self.df.iloc[i]["type"],
                    "genre": self.df.iloc[i]["listed_in"],
                    "similarity_score": float(score),
                }
            )

        return recommendations

    def save_model(self, path):
        """Save model artifacts"""
        with open(f"{path}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        self.df.to_csv(f"{path}/processed_data.csv", index=False)
