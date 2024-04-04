# Dataset
import pandas as pd
import numpy as np

# Util
import ast
from nltk.stem.snowball import SnowballStemmer

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# My scripts
from Src.scripts.approaches.filters.popularity_rankings import PopularityRanking

"""
    References:
        - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""

class ContentBasedFiltering:
    """
        ContentBasedFiltering - Movie Recommendation System Based on Content-Based Filtering.

        This class provides functions for implementing a movie recommendation system based on content-based filtering.
        It includes methods for recommending movies based on movie descriptions and metadata (cast, crew, keywords, and
        genre).
    """

    @staticmethod
    def description_based_recommender(tabular_dataset_handler, movie_title, N, improved=False):
        """
            Recommends the N top most similar movies to the target movie based on movie descriptions and taglines.

            Therefore, this function does not provide personalized recommendations based on the user.

            Parameters:
                tabular_dataset_handler (TabularDatasetHandler): An instance of the DatasetHandler class.
                movie_title (str): The title of the movie.
                N (int): The number of top most similar movies to recommend.
                improved (bool): Whether to use improved recommendation using the weighted rating (wr) when True.

            Returns:
                pd.DataFrame: The N top most similar movies based on content.
        """
        # Initialize a brand-new copy of the 'movies_df'
        movies_df = tabular_dataset_handler.get_small_movies_df_deepcopy()

        # Process the 'tagline' column
        movies_df['tagline'] = movies_df['tagline'].fillna('')

        # Add a new column 'description' containing the concatenation of the 'overview' and of the 'tagline' columns
        movies_df['description'] = movies_df['overview'] + movies_df['tagline']
        movies_df['description'] = movies_df['description'].fillna('')

        # Convert the 'description' of each movie into a matrix of token counts
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
        tf_idf_matrix = tf.fit_transform(movies_df['description'])

        # Compute the cosine similarities between each couple of movies
        # Note: computing the cosine similarity allows to denote the similarity between two movies.
        # Note: since we have used the TF-IDF vectorizer, calculating the dot product will directly give us the cosine
        # similarity score between each couple of movies.
        cos_similarity = linear_kernel(tf_idf_matrix, tf_idf_matrix)

        # Map movie titles to their indices in the 'movies_df'
        #movies_df = movies_df.drop(columns='level_0', errors='ignore')  # Drop the existing index column if it exists
        movies_df = movies_df.reset_index()
        titles = movies_df['title']
        indices = pd.Series(movies_df.index, index=movies_df['title'])

        if not improved:
            return ContentBasedFiltering.__get_recommendations(
                movies_df, movie_title, N, cos_similarity, titles, indices
            )[['id', 'title', 'year', 'description']]
        else:
            return ContentBasedFiltering.__get_improved_recommendations(
                movies_df, movie_title, N, cos_similarity, titles, indices
            )[['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'wr']]

    @staticmethod
    def metadata_based_recommender(tabular_dataset_handler, movie_title, N, improved=False):
        """
            Recommends the N top most similar movies to the target movie based on movie metadata (cast, crew, keywords,
            and genre).

            Therefore, this function does not provide personalized recommendations based on the user.

            Parameters:
                tabular_dataset_handler (TabularDatasetHandler): An instance of the DatasetHandler class.
                movie_title (str): The title of the movie.
                N (int): The number of top most similar movies to recommend.
                improved (bool): Whether to use improved recommendation using the weighted rating (wr) when True.

            Returns:
                pd.DataFrame: The N top most similar movies based on content.
        """
        # Initialize some brand-new copies of the required dataframes from dataset
        movies_df = tabular_dataset_handler.get_small_movies_df_deepcopy()
        credits_df = tabular_dataset_handler.get_credits_df_deepcopy()
        keywords_df = tabular_dataset_handler.get_keywords_df_deepcopy()

        # Add the columns 'cast', 'crew', 'keywords' (from the corresponding dataframes) to the 'movies_df'
        movies_df = movies_df.merge(credits_df, on='id')
        movies_df = movies_df.merge(keywords_df, on='id')

        # Process 'cast' and 'keywords' columns
        movies_df['cast'] = movies_df['cast'].apply(ast.literal_eval)
        movies_df['crew'] = movies_df['crew'].apply(ast.literal_eval)
        movies_df['keywords'] = movies_df['keywords'].apply(ast.literal_eval)

        # Add the columns 'cast_size' and 'crew_size' to the 'movies_df'
        movies_df['cast_size'] = movies_df['cast'].apply(lambda x: len(x))
        movies_df['crew_size'] = movies_df['crew'].apply(lambda x: len(x))

        # Add the column 'director' to the 'movies_df'
        movies_df['director'] = movies_df['crew'].apply(ContentBasedFiltering.__get_director)

        # Process the 'cast' column, for extracting only the names of the actors
        movies_df['cast'] = movies_df['cast'].apply(
            lambda cast_row: [i['name'] for i in cast_row]
            if isinstance(cast_row, list) else []
        )
        # Process the 'cast' column, for extracting only the 3 top actors that appear in the credits list for each movie
        movies_df['cast'] = movies_df['cast'].apply(
            lambda cast_row: cast_row[:3]
            if len(cast_row) >= 3 else cast_row
        )

        # Process the 'keywords' column, for extracting only the names of the keywords
        movies_df['keywords'] = movies_df['keywords'].apply(
            lambda keyword_row: [i['name'] for i in keyword_row]
            if isinstance(keyword_row, list) else []
        )

        # Process the 'cast' column, for stripping spaces (so that NameSurname is recognised as the same actor)
        movies_df['cast'] = movies_df['cast'].apply(
            lambda cast_row: [str.lower(actor.replace(" ", "")) for actor in cast_row]
        )

        # Process the 'director' column, for mentioning each director 3 times to give it more weight (w.r.t. to the entire cast)
        movies_df['director'] = movies_df['director'].astype('str').apply(
            lambda director: str.lower(director.replace(" ", ""))
        )
        movies_df['director'] = movies_df['director'].apply(
            lambda director: [director, director, director]
        )

        # Create a serie 'keyword_s' of movies' keywords
        keyword_s = (movies_df
                     .apply(lambda row: pd.Series(row['keywords']), axis=1)
                     .stack()
                     .reset_index(level=1, drop=True)
        )
        keyword_s.name = 'keyword'

        # Process the 'keyword_s' serie, by calculating the frequency counts of every keyword
        keyword_s = keyword_s.value_counts()

        # Remove keywords that occur only once
        keyword_s = keyword_s[keyword_s > 1]

        # Convert every keyword word to its stem (so that "similar" words are considered the same)
        stemmer = SnowballStemmer('english')

        movies_df['keywords'] = movies_df['keywords'].apply(
            lambda keyword : ContentBasedFiltering.__filter_keywords(keyword, keyword_s)
        )
        movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])

        # Process the 'keywords' column, for stripping spaces
        movies_df['keywords'] = movies_df['keywords'].apply(
            lambda x: [str.lower(i.replace(" ", "")) for i in x]
        )

        # Add the column 'soup' to the 'movies_df', containing metadata dump for every movie
        movies_df['soup'] = (movies_df['keywords'] +
                             movies_df['cast'] +
                             movies_df['director'] +
                             movies_df['genres']
        )
        movies_df['soup'] = movies_df['soup'].apply(lambda x: ' '.join(x))

        # Convert the 'soup' of each movie into a matrix of token counts
        count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
        count_matrix = count.fit_transform(movies_df['soup'])

        # Compute the cosine similarities between each couple of movies
        cos_similarity = cosine_similarity(count_matrix, count_matrix)

        # Map movie titles to their indices in the 'movies_df'
        #movies_df = movies_df.drop(columns='level_0', errors='ignore')  # Drop the existing index column if it exists
        movies_df = movies_df.reset_index()
        titles = movies_df['title']
        indices = pd.Series(movies_df.index, index=movies_df['title'])

        if not improved:
            return ContentBasedFiltering.__get_recommendations(
                movies_df, movie_title, N, cos_similarity, titles, indices
            )[['id', 'title', 'year', 'soup']]
        else:
            return ContentBasedFiltering.__get_improved_recommendations(
                movies_df, movie_title, N, cos_similarity, titles, indices
            )[['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'wr']]

    @staticmethod
    def __get_recommendations(movies_df, movie_title, N, cos_similarity, titles, indices):
        """
            Returns the N most similar movies based on the cosine similarity score.

            Parameters:
                movies_df (pd.DataFrame): DataFrame containing movie data.
                movie_title (str): Title of the target movie.
                N (int): Number of movies to recommend.
                cos_similarity (np.array): Cosine similarity matrix.
                titles (pd.Series): Series of movie titles.
                indices (pd.Series): Series mapping titles to their indices.

            Returns:
                pd.DataFrame: The N most similar movies.

        """
        idx = indices[movie_title]
        sim_scores = list(enumerate(cos_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:N]

        movies_indices = [i[0] for i in sim_scores]

        return movies_df.loc[movies_indices]

    @staticmethod
    def __get_improved_recommendations(movies_df, movie_title, N, cos_similarity, titles, indices, percentile=0.60):
        """
            Returns improved recommendations by filtering bad movies and considering popularity and critical response.

            To do that it calculates the top 25 movies based on similarity scores and calculate the i-th percentile
            movies, that is used as value m for calculating the weighted rating of each movie using the IMDb's formula.

            Parameters:
                movies_df (pd.DataFrame): DataFrame containing movie data.
                movie_title (str): Title of the target movie.
                N (int): Number of movies to recommend.
                cos_similarity (np.array): Cosine similarity matrix.
                titles (pd.Series): Series of movie titles.
                indices (pd.Series): Series mapping titles to their indices.
                percentile (float): Percentile value for calculating weighted rating.

            Returns:
                pd.DataFrame: Improved N most similar movies.

        """
        # Use __get_recommendations to get the recommended movies
        recommended_movies_df = ContentBasedFiltering.__get_recommendations(movies_df, movie_title, N, cos_similarity, titles, indices)

        # Create a 'new_movies_df' starting from the 'recommended_movies_df', extracting only the interesting columns
        new_movies_df = recommended_movies_df[
            ['id', 'title', 'year', 'genres', 'vote_count', 'vote_average', 'popularity']
        ]

        # Reset the index to ensure 'title' is a regular column
        #new_movies_df = new_movies_df.reset_index(drop=True)

        # Use __top_movies_IMDB_wr_formula to improve the recommendation
        return PopularityRanking._top_movies_IMDB_wr_formula(new_movies_df, N, percentile)

    @staticmethod
    def __get_director(crew_row):
        """
            Returns the director from the 'crew' row.

            Parameters:
                crew_row (list): List of crew members for a movie.

            Returns:
                str or np.nan: Director's name or NaN if not found.
        """
        for x in crew_row:
            if x['job'] == 'Director':
                return x['name']

        return np.nan

    @staticmethod
    def __filter_keywords(keyword_s, keyword_list):
        """
            Models the keywords of a movie that are not present in 'keyword_s' and returns the other ones.

            Parameters:
                keyword_s (pd.Series): Movie keywords.
                keyword_list (pd.Series): Frequency counts of every keyword.

            Returns:
                list: List of filtered keywords.
        """
        words = []

        for k in keyword_list:
            if k in keyword_s:
                words.append(k)

        return words
