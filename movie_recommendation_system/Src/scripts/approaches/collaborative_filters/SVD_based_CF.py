# Util
import os
from overrides import overrides

# Machine learning
from surprise import Reader, Dataset, SVD, dump
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

# My scripts
from Src.scripts.approaches.filters.collaborative_filtering import CollaborativeFilteringInterface

"""
    References:
        - https://www.kaggle.com/code/rounakbanik/movie-recommender-systems
"""


class SVD_Based_CollaborativeFilter(CollaborativeFilteringInterface):
    """
        SVD_Based_CollaborativeFilter - User-Based Collaborative Filtering using SVD Model

        This class implements a movie recommendation system based on user-based collaborative filtering. It uses the SVD
        (Singular Value Decomposition) model from the surprise library to provide personalized movie recommendations for
        users.

        Parameters:
            tabular_dataset_handler (TabularDatasetHandler): An instance of TabularDatasetHandler.

        Attributes:
            __users_ratings_df (pd.DataFrame): A copy of the users ratings dataframe from the dataset.
            __movies_df (pd.DataFrame): A copy of the small movies dataframe from the dataset.
            __dataset (surprise.dataset.Dataset): Surprise dataset created from users ratings dataframe.
            __model (surprise.prediction_algorithms.matrix_factorization.SVD): SVD model for collaborative filtering.
            __trainset (surprise.Trainset): Training set for the SVD model.
            __testset (list): Test set for evaluating the model performance.
            __models_path (str): Path to store trained models.
            __trained_model_filename (str): Filename of the trained SVD model.
    """

    def __init__(self, tabular_dataset_handler):
        super().__init__()

        # Initialize some brand-new copies of the required dataframes from dataset
        self.__users_ratings_df = tabular_dataset_handler.get_users_ratings_df_deepcopy()
        self.__movies_df = tabular_dataset_handler.get_small_movies_df_deepcopy()

        # Create a dataset from the 'users_ratings_df' for training and test the recommender system models provided by
        # the 'surprise' library
        reader = Reader()
        self.__dataset = Dataset.load_from_df(self.__users_ratings_df[['userId', 'movieId', 'rating']], reader)

        # Initialize the model, the trainset and the testset
        self.__model = None
        self.__trainset = None
        self.__testset = None

        # Model storage settings
        self.__models_path = "..\\Src\\trained_models"
        self.__trained_model_filename = None

    @overrides
    def train(self):
        """
            Train the SVD model on the training set and store the trained model.
        """
        # Initialize the SVD model
        self.__model = SVD()

        # Split the dataset into train and test sets
        trainset, testset = train_test_split(self.__dataset, test_size=0.2)

        # Train the SVD model on the train set
        self.__model.fit(trainset)

        # Store the trained model
        model_filename = "SVD_model.pkl"
        self.__trained_model_filename = os.path.join(self.__models_path, model_filename)
        dump.dump(self.__trained_model_filename, algo=self.__model)

        # Save the test set for later evaluation
        self.__testset = testset

    @overrides
    def evaluate_performance(self):
        """
            Evaluate the performance of the trained model on the test set using RMSE and MAE metrics.

            Returns:
                rmse (float): Root Mean Squared Error.
                mae (float): Mean Absolute Error.
        """
        # Check whether the model is valid
        self.__model_check()

        # Make predictions on the saved test set
        predictions = self.__model.test(self.__testset)

        # Evaluate RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)

        return rmse, mae

    def train_with_cross_validation(self):
        """
            Train the SVD model using cross-validation and store the trained model.
        """
        # Initialize the SVD model
        self.__model = SVD()

        # Train the SVD model
        self.__trainset = self.__dataset.build_full_trainset()
        self.__model.fit(self.__trainset)

        # Store the trained model
        model_filename = "SVD_with_cross_validation_model.pkl"
        self.__trained_model_filename = os.path.join(self.__models_path, model_filename)
        dump.dump(self.__trained_model_filename, algo=self.__model)

    def evaluate_performance_with_cross_validation(self):
        """
            Evaluate the performance of the model with cross-validation using RMSE and MAE metrics.

            Returns:
                rmse (float): Mean RMSE across folds.
                mae (float): Mean MAE across folds.
        """
        # Check whether the model is valid
        self.__model_check()

        # Perform cross-validation, for evaluating RMSE and MAE
        results = cross_validate(self.__model, self.__dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)

        # Extract mean RMSE and mean MAE
        rmse = sum(results['test_rmse']) / len(results['test_rmse'])
        mae = sum(results['test_mae']) / len(results['test_mae'])

        return rmse, mae

    @overrides
    def predict(self, user_id, movie_id, gt_rating=None):
        """
            Make a movie rating prediction for a user on a given movie.

            Parameters:
                user_id (int): User ID for prediction.
                movie_id (int): Movie ID for prediction.
                gt_rating (float, optional): Ground truth rating (if available).

            Returns:
                prediction (surprise.prediction_algorithms.predictions.Prediction): Prediction result.
        """
        # Check whether the model is valid
        self.__model_check()

        if gt_rating is not None:
            return self.__model.predict(user_id, movie_id, gt_rating)
        else:
            return self.__model.predict(user_id, movie_id)

    def __model_check(self):
        """
            Check whether the SVD model has been trained and is available to be preloaded.
        """
        # If the model has not been trained yet
        if self.__model == None:
            # If a pretrained model exists in memory, load it
            if os.path.exists(self.__trained_model_filename):
                self.__model, self.__algo, self.__trainset = dump.load(self.__trained_model_filename)
            else:
                raise FileNotFoundError(f"The model file '{self.__trained_model_filename}' does not exist.")