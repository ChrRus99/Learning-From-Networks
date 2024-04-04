# Dataset
import pandas as pd
import copy

# Util
import os
import ast
import numpy as np

"""
    References:
        - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class TabularDatasetHandler:
    """
        A class to handle tabular datasets related to movie recommendations.

        Datasets:
            - movies_metadata.csv: Contains information on 45,000 movies featured in the Full MovieLens dataset.
            - keywords.csv: Contains movie plot keywords in the form of a stringified JSON Object.
            - credits.csv: Consists of Cast and Crew Information for all movies.
            - links.csv: Contains the TMDB and IMDB IDs of all movies in the Full MovieLens dataset.
            - links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies.
            - ratings_small.csv: Subset of 100,000 ratings from 700 users on 9,000 movies.
    """

    def __init__(self):
        data_path = "..\\Src\\datasets\\raw"

        loader = self.__DatasetLoader(data_path)
        (
            self.__users_ratings_df,
            self.__movies_df,
            self.__links_df,
            self.__credits_df,
            self.__keywords_df,
        ) = loader.load_datasets()

    def get_users_ratings_df_deepcopy(self):
        """
            Returns a deep copy of the users_ratings DataFrame.

            Returns:
                pd.DataFrame: A deep copy of the users_ratings DataFrame.
        """
        return copy.deepcopy(self.__users_ratings_df)

    def get_movies_df_deepcopy(self):
        """
            Returns a deep copy of the movies DataFrame.

            Returns:
                pd.DataFrame: A deep copy of the movies DataFrame.
        """
        return copy.deepcopy(self.__movies_df)

    def get_small_movies_df_deepcopy(self):
        """
            Returns a deep copy of a subset of movies DataFrame containing only those with valid 'tmdbId' and rated by
            at least one user.

            Returns:
                pd.DataFrame: A deep copy of the subset of movies DataFrame.
        """
        # Discard the movies that have no 'tmdbId'
        new_links_df = self.__links_df[self.__links_df['tmdbId'].notnull()]['tmdbId'].astype('int')

        # Discard all movies that are not rated by any user
        small_movies_df = self.__movies_df[self.__movies_df['id'].isin(new_links_df)]
        return small_movies_df

    def get_links_df_deepcopy(self):
        """
            Returns a deep copy of the links DataFrame.

            Returns:
                pd.DataFrame: A deep copy of the links DataFrame.
        """
        return copy.deepcopy(self.__links_df)

    def get_credits_df_deepcopy(self):
        """
            Returns a deep copy of the credits DataFrame.

            Returns:
                pd.DataFrame: A deep copy of the credits DataFrame.
        """
        return copy.deepcopy(self.__credits_df)

    def get_keywords_df_deepcopy(self):
        """
            Returns a deep copy of the keywords DataFrame.

            Returns:
                pd.DataFrame: A deep copy of the keywords DataFrame.
        """
        return copy.deepcopy(self.__keywords_df)

    def preprocess_datasets(self):
        """
            Preprocesses the loaded datasets.

            This method should be called after loading the datasets to perform necessary preprocessing steps.
        """
        preprocessor = self.__DatasetPreprocessor(
            self.__users_ratings_df,
            self.__movies_df,
            self.__links_df,
            self.__credits_df,
            self.__keywords_df,
        )

        preprocessor.preprocess_datasets()

        self.__users_ratings_df = preprocessor.users_ratings_df
        self.__movies_df = preprocessor.movies_df
        self.__links_df = preprocessor.links_df
        self.__credits_df = preprocessor.credits_df
        self.__keywords_df = preprocessor.keywords_df

    class __DatasetLoader:
        """
            A private class responsible for loading datasets.
        """

        def __init__(self, data_path):
            self.__data_path = data_path

        def load_datasets(self):
            """
                Loads datasets from the specified data path.

                Returns:
                    Tuple: A tuple containing DataFrames for users_ratings, movies, links, credits, and keywords.
            """
            # Data paths
            users_ratings_data_path = os.path.join(self.__data_path, "ratings_small.csv")   # "ratings.csv" --> non fitta in memoria
            movies_data_path = os.path.join(self.__data_path, "movies_metadata.csv")
            links_small_data_path = os.path.join(self.__data_path, "links_small.csv")
            credits_data_path = os.path.join(self.__data_path, "credits.csv")
            keywords_data_path = os.path.join(self.__data_path, "keywords.csv")

            # Data loading
            try:
                users_ratings_df = pd.read_csv(users_ratings_data_path)
                movies_df = pd.read_csv(movies_data_path)
                links_df = pd.read_csv(links_small_data_path)
                credits_df = pd.read_csv(credits_data_path)
                keywords_df = pd.read_csv(keywords_data_path)
            except Exception as e:
                raise Exception(f"Error loading datasets: {e}")

            return users_ratings_df, movies_df, links_df, credits_df, keywords_df

    class __DatasetPreprocessor:
        """
            A private class responsible for preprocessing datasets.
        """

        def __init__(self, users_ratings_df, movies_df, links_df, credits_df, keywords_df):
            self.users_ratings_df = users_ratings_df
            self.movies_df = movies_df
            self.links_df = links_df
            self.credits_df = credits_df
            self.keywords_df = keywords_df

        def preprocess_datasets(self):
            """
                Preprocesses the loaded datasets in place.
            """
            self.__preprocess_users_ratings_df()
            self.__preprocess_movies_df()
            self.__preprocess_links_df()
            self.__preprocess_credits_df()
            self.__preprocess_keywords_df()

        def __preprocess_users_ratings_df(self):
            """
                Preprocesses the users_ratings DataFrame.
            """
            pass

        def __preprocess_movies_df(self):
            """
                Preprocesses the movies DataFrame.
            """
            # Process the 'genres' column
            self.movies_df['genres'] = (self.movies_df['genres']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda genres: [genre['name'] for genre in genres] if isinstance(genres, list) else [])
            )

            # Add a new column 'year'
            self.movies_df['year'] = (pd.to_datetime(self.movies_df['release_date'], errors='coerce')
                .apply(lambda date: str(date).split('-')[0] if date != np.nan else np.nan)
            )

            # Remove bad 'id'
            self.movies_df = self.movies_df.drop([19730, 29503, 35587])

            # Process the 'id' column
            self.movies_df['id'] = pd.to_numeric(   # Handle non-numeric values by converting them to NaN
                self.movies_df['id'],
                errors='coerce'
            )
            self.movies_df = self.movies_df.dropna( # Drop rows with NaN values in the 'id' column
                subset=['id']
            )
            self.movies_df['id'] = self.movies_df['id'].astype('int')

            # Process the 'production_companies' column
            self.movies_df['production_companies'] = (self.movies_df['production_companies']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda prod_companies: [prod_company['name'] for prod_company in prod_companies]
                    if isinstance(prod_companies, list) else [])
            )

            # Process the 'production_countries' column
            self.movies_df['production_countries'] = (self.movies_df['production_countries']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda prod_countries: [prod_country['name'] for prod_country in prod_countries]
                    if isinstance(prod_countries, list) else [])
            )

            # Process the 'belongs_to_collection' column
            self.movies_df['belongs_to_collection'] = (self.movies_df['belongs_to_collection']
                .fillna('[]')
                .apply(ast.literal_eval)
                .apply(lambda collections: collections.get('name') if isinstance(collections, dict) else None)
            )

        def __preprocess_links_df(self):
            """
                Preprocesses the links DataFrame.
            """
            # Process the 'tmdbId' column
            self.links_df.dropna(subset=['tmdbId'], inplace=True)
            self.links_df['tmdbId'] = self.links_df['tmdbId'].astype('int')

        def __preprocess_credits_df(self):
            """
                Preprocesses the credits DataFrame.
            """
            # Process the 'id' column
            self.credits_df['id'] = self.credits_df['id'].astype('int')

        def __preprocess_keywords_df(self):
            """
                Preprocesses the keywords DataFrame.
            """
            # Process the 'id' column
            self.keywords_df['id'] = self.keywords_df['id'].astype('int')