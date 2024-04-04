from abc import ABC, abstractmethod
import os


class CollaborativeFilteringInterface(ABC):
    """
        CollaborativeFilteringInterface - Abstract Class for Collaborative Filtering Movie Recommendation System.

        This abstract class defines the structure of a movie recommendation system based on collaborative-filtering.

        Attributes:
            _dataset: Dataset for collaborative filtering.
            _model: Model for collaborative filtering.
            _algo: Algorithm for collaborative filtering.
            _trainset: Trainset for collaborative filtering.
            _models_path: Path to store trained models.
            _trained_model_path: Full path for storing the trained model.
    """
    def __init__(self, datset=None, model=None, algo=None, trainset=None):
        self._dataset = datset
        self._model = model
        self._algo = algo
        self._trainset = trainset

        self._models_path = "..\\Src\\models"
        model_filename = "model"
        self._trained_model_path = os.path.join(self._models_path, model_filename)

    @abstractmethod
    def train(self):
        """
            Abstract method for training the collaborative filtering model.
        """
        pass

    @abstractmethod
    def evaluate_performance(self):
        """
            Abstract method for evaluating the performance of the collaborative filtering model.

            Returns:
                Tuple: Evaluation metrics MAE and MRSE.
        """
        pass

    @abstractmethod
    def predict(self, user_id, movie_id, gt_rating = None):
        """
            Abstract method for making predictions using the collaborative filtering model.

            Parameters:
                user_id: User ID for prediction.
                movie_id: Movie ID for prediction.
                gt_rating (float, optional): Ground truth rating (if available).

            Returns:
                Prediction result (specific to the implementation).
        """
        pass