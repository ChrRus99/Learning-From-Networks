# Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Others
from contextlib import redirect_stdout
from io import StringIO

# My scripts
from Src.scripts.data.tabular_dataset_handler import TabularDatasetHandler
from Src.scripts.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from Src.scripts.approaches.filters.content_based_filtering import ContentBasedFiltering
from Src.scripts.approaches.filters.collaborative_filtering import CollaborativeFilteringInterface
from Src.scripts.approaches.collaborative_filters.SVD_based_CF import SVD_Based_CollaborativeFilter
from Src.scripts.approaches.collaborative_filters.GNN_based_CF import GNN_Based_CollaborativeFilter

"""
    References:
        - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class HybridFiltering:
    """
        HybridFiltering - A Movie Recommendation System using Hybrid Filtering approach

        This class implements a movie recommendation system based on a hybrid filtering approach. It combines
        collaborative filtering using collaborative filtering models and content-based filtering to provide personalized
        movie recommendations for users.

        Parameters:
            tabular_dataset_handler (TabularDatasetHandler): An instance of TabularDatasetHandler.
            model (CollaborativeFilteringInterface): The collaborative filtering model.

        Attributes:
            __tdh (TabularDatasetHandler): Instance of TabularDatasetHandler.
            __small_movies_df (pd.DataFrame): A copy of the small movies dataframe from the dataset.
            __links_df (pd.DataFrame): A copy of the links dataframe from the dataset.
            __users_ratings_df (pd.DataFrame): A copy of the users ratings dataframe from the dataset.
            __id_map_df (pd.DataFrame): A dataframe mapping movie titles to their corresponding IDs.
            __indices_map (pd.DataFrame): A dataframe mapping movie IDs to their corresponding indices.
            __model (CollaborativeFilteringInterface): The collaborative filtering model (SVD or GNN).
            __gdh (HeterogeneousGraphDatasetHandler): Instance of HeterogeneousGraphDatasetHandler for GNN-based CF.
    """

    def __init__(self, tabular_dataset_handler: TabularDatasetHandler, model: CollaborativeFilteringInterface):
        self.__tdh = tabular_dataset_handler

        # Initialize some brand-new copies of the required dataframes from dataset
        self.__small_movies_df = tabular_dataset_handler.get_small_movies_df_deepcopy()
        self.__links_df = tabular_dataset_handler.get_links_df_deepcopy()
        self.__users_ratings_df = tabular_dataset_handler.get_users_ratings_df_deepcopy()

        # Process the 'links_df' dataframe
        self.__links_df = self.__links_df[['movieId', 'tmdbId']]
        self.__links_df['tmdbId'] = self.__links_df['tmdbId'].apply(HybridFiltering.__convert_int)
        self.__links_df.columns = ['movieId', 'id']

        # Join operation between 'links_df' and 'small_movies_df' on tmdbId index
        self.__id_map_df = self.__links_df.merge(self.__small_movies_df[['title', 'id']], on='id').set_index('title')
        self.__indices_map = self.__id_map_df.set_index('id')

        # Handle the different possible collaborative filter models and their training
        self.__model = None
        self.__gdh = None

        if isinstance(model, SVD_Based_CollaborativeFilter):
            self.__model = model
            self.__model.train()
        elif isinstance(model, GNN_Based_CollaborativeFilter):
            self.__model = model
            # Suppress prints
            with StringIO() as fake_stdout:
                with redirect_stdout(fake_stdout):
                    self.__model.train()
            self.__gdh = HeterogeneousGraphDatasetHandler(tabular_dataset_handler)

    def predict(self, user_id, movie_title, N):
        """
            Suggests the target user N movies similar to the selected target movie, based on his/her tastes (i.e., on
            user's past ratings).

            The predictions are based on a hybrid approach combining collaborative and content-based filtering.

            Parameters:
                user_id (int): The ID of the user.
                movie_title (str): The title of the target movie.
                N (int): The number of movie recommendations to return.

            Returns:
                pd.DataFrame: DataFrame containing movie recommendations.
        """
        # Get the movies suggestion of the metadata_based_recommender
        new_movies_df = ContentBasedFiltering.metadata_based_recommender(self.__tdh, movie_title, N, improved=False)
        new_movies_df = new_movies_df  # [['title', 'vote_count', 'vote_average', 'year', 'id']]

        # Predict the N top most similar movies sorted on the basis of expected ratings by that particular user
        if isinstance(self.__model, SVD_Based_CollaborativeFilter):
            # Do prediction on 'new_movies_df'
            new_movies_df['est_rating'] = new_movies_df['id'].apply(
                lambda x: self.__model.predict(user_id, self.__indices_map.loc[x]['movieId']).est
            )
        elif isinstance(self.__model, GNN_Based_CollaborativeFilter):
            # Update the graph dataset with new edges to allow the prediction on new movies (not yet rated by user)
            # self.__preprocess_gdh(user_id, self.__users_ratings_df, new_movies_df),

            # Do prediction on 'new_movies_df'
            new_movies_df['est_rating'] = new_movies_df['id'].apply(
                lambda x: self.__model.predict(user_id, self.__indices_map.loc[x]['movieId'])[0]
            )
        new_movies_df = new_movies_df.sort_values('est_rating', ascending=False)

        # Add ground truth ratings to the output
        new_movies_df['gt_rating'] = new_movies_df['id'].apply(
            lambda x: self.__users_ratings_df[
                (self.__users_ratings_df['userId'] == user_id) &
                (self.__users_ratings_df['movieId'] == self.__indices_map.loc[x]['movieId'])
                ]['rating'].values[0]
            if ((user_id in self.__users_ratings_df['userId'].values) and
                (self.__indices_map.loc[x]['movieId'] in self.__users_ratings_df['movieId'].values) and
                (len(self.__users_ratings_df[
                         (self.__users_ratings_df['userId'] == user_id) &
                         (self.__users_ratings_df['movieId'] == self.__indices_map.loc[x]['movieId'])
                ]) > 0))
            else np.nan
        )

        return new_movies_df

    def __preprocess_gdh(self, user_id, users_ratings_df, movies_df):
        """
            This function is paramount for GNN-based collaborative filtering.

            Creates an edge in the heterogeneous graph dataset for each movie the user has not seen yet.
            This is necessary in order for the GNN collaborative filter approach to be applicable.

            Parameters:
                user_id (int): The ID of the user.
                users_ratings_df (pd.DataFrame): The user dataframe containing user ratings.
                movies_df (pd.DataFrame): The movie dataframe containing movie information.
        """
        # Extract seen movies for the given user
        seen_movies = set(users_ratings_df[users_ratings_df['userId'] == user_id]['movieId'])

        # Get the list of all movie IDs
        all_movie_ids = set(movies_df['id'])

        # Find the movies not rated by the user
        missing_movies = all_movie_ids - seen_movies

        ratings = []

        # Create rating record for each movie not rated by the user
        for movie_id in tqdm(missing_movies, desc="Adding possible missing edges to do prediction using GNNs", unit="movie"):
            ratings.append({'movieId': movie_id, 'rating': None, 'userId': user_id})

        # Add the rating records to the graph dataset handler
        self.__gdh.add_users_ratings_data(pd.DataFrame.from_records(ratings))

    @staticmethod
    def __convert_int(x):
        try:
            return int(x)
        except:
            return np.nan