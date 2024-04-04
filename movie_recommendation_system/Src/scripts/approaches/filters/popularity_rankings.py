# Dataset
import pandas as pd

"""
    References:
        - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class PopularityRanking:
    """
        PopularityRanking - Movie Recommendation System Based on Popularity Rankings.

        This class implements a simple movie recommendation system based on popularity rankings.
        It provides methods to retrieve the top movies according to IMDB's weighted rating formula and the top movies
        for a specific genre.

    """

    @staticmethod
    def top_movies_IMDB_wr_formula(tabular_dataset_handler, N):
        """
            Returns the top N most popular movies according to IMDB's weighted rating formula.

            Therefore, this function does not provide personalized recommendations based on the user.

            Parameters:
                tabular_dataset_handler (TabularDatasetHandler): An instance of the DatasetHandler class.
                N (int): The number of top movies to recommend.

            Returns:
                pd.DataFrame: The top N movies based on IMDB's weighted rating ('wr').
        """

        # Initialize a brand-new copy of the 'movies_df'
        movies_df = tabular_dataset_handler.get_movies_df_deepcopy()

        return PopularityRanking._top_movies_IMDB_wr_formula(movies_df, N)

    @staticmethod
    def top_movies_for_genre(tabular_dataset_handler, genre, N):
        """
            Returns the top N most popular movies of the specified genre according to IMDB's weighted rating formula.

            Therefore, this function does not provide personalized recommendations based on the user.

            Parameters:
                tabular_dataset_handler (TabularDatasetHandler): An instance of the DatasetHandler class.
                genre (str): The genre of the recommended movies.
                N (int): The number of top movies to recommend for the specified genre.

            Returns:
                pd.DataFrame: The top N movies of the specified genre based on IMDB's weighted rating ('wr').
        """

        # Initialize a brand-new copy of the 'movies_df'
        movies_df = tabular_dataset_handler.get_movies_df_deepcopy()

        # Create a new 'genre_movies_df' dataframe equivalent to the 'movies_df', where each movie record is replicated
        # m times, with m = number of movie's genre, in such a way to have only one of genre for each replicated record
        # of that movie
        genres_s = movies_df \
            .apply(lambda row: pd.Series(row['genres']), axis=1) \
            .stack() \
            .reset_index(level=1, drop=True)
        genres_s.name = 'genres'

        genre_movies_df = movies_df.drop('genres', axis=1).join(genres_s)

        # Extract only the movies that are of the required genre
        genre_movies_df = genre_movies_df[genre_movies_df['genres'] == genre]

        return PopularityRanking._top_movies_IMDB_wr_formula(genre_movies_df, N).rename(columns={'genres': 'genre'})

    @staticmethod
    def _top_movies_IMDB_wr_formula(movies_df, N, percentile=0.85):
        """
            Returns the top N movies based on IMDB's weighted rating formula.

            Parameters:
                movies_df (pd.DataFrame): DataFrame containing movie data.
                N (int): The number of top movies to recommend.
                percentile (float): Percentile value for calculating weighted rating.

            Returns:
                pd.DataFrame: The top N movies based on IMDB's weighted rating ('wr').
        """
        # Extract 'vote_counts' and 'vote_averages'
        vote_counts = movies_df[movies_df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = movies_df[movies_df['vote_average'].notnull()]['vote_average'].astype('int')

        # Mean vote across the whole report
        C = vote_averages.mean()

        # Minimum vote required to be listed in the chart
        m = vote_counts.quantile(percentile)

        # Create a new dataframe with only the movies qualified to be considered
        qualified_movies_df = movies_df[
            (movies_df['vote_count'] >= m) &
            movies_df['vote_count'].notnull() &
            movies_df['vote_average'].notnull()
            ][['id', 'title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

        qualified_movies_df['vote_count'] = qualified_movies_df['vote_count'].astype(int)
        qualified_movies_df['vote_average'] = qualified_movies_df['vote_average'].astype(int)

        # Add a new column weighted rating (wr) to the qualified_movies_df
        qualified_movies_df['wr'] = qualified_movies_df.apply(
            lambda movie: PopularityRanking._weighted_rating(movie, C, m),
            axis=1
        )

        # Extract and sort the first 250 qualified movies per weighted rating ('wr')
        qualified_movies_df = qualified_movies_df \
            .sort_values('wr', ascending=False) \
            .head(N)

        return qualified_movies_df

    @staticmethod
    def _weighted_rating(movie, C, m):
        """
            Computes the IMDB weighted rating for a movie.

            Parameters:
                movie (pd.Series): Series representing a movie's data.
                C (float): Mean vote across the whole report.
                m (float): Minimum vote required to be listed in the chart.

            Returns:
                float: IMDB's weighted rating for the movie.
        """
        # Number of votes for the movie
        v = movie['vote_count']
        # Average rating for the movie
        R = movie['vote_average']

        # Compute the IMDB weighted rating for the movie
        return (v / (v + m) * R) + (m / (v + m) * C)
