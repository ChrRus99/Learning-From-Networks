# Dataset
import pandas as pd
import numpy as np

# Util
import os
import shutil
from overrides import overrides
from typing import Tuple

# Pytorch geometric
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

# Machine learning

# My scripts
from Src.scripts.data.graph_dataset_handler import HeterogeneousGraphDatasetHandler
from Src.scripts.approaches.filters.collaborative_filtering import CollaborativeFilteringInterface
from Src.scripts.approaches.models.GNN_regression_model import GNNEncoderInterface, Model

"""
   References:
        - Link Regression on MovieLens:
            https://colab.research.google.com/drive/1N3LvAO0AXV4kBPbTMX866OwJM9YS6Ji2?usp=sharing
        - Link Prediction on MovieLens:
            https://colab.research.google.com/drive/1xpzn1Nvai1ygd_P5Yambc_oe4VBPK_ZT?usp=sharing
        - Link Prediction on Heterogeneous Graphs with PyG:
            https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70
        - Heterogeneous Graph Learning (Pytorch-geometric):
            https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
        - GRAPH Link Prediction w/ DGL on Pytorch and PyG Code Example | GraphML | GNN:
            https://www.youtube.com/watch?v=wxJ84sMJfUA&ab_channel=code_your_own_AI
        - PyTorch Geometric tutorial: Graph Autoencoders & Variational Graph Autoencoders:
            https://www.youtube.com/watch?v=qA6U4nIK62E&ab_channel=AntonioLonga
"""


class GNN_Based_CollaborativeFilter(CollaborativeFilteringInterface):
    """
        GNN_Based_CollaborativeFilter is a class implementing a movie recommendation system using collaborative filtering.
        Specifically, it is a user-based collaborative filter that makes use of GNNs models.

        Attributes:
            __gdh (HeterogeneousGraphDatasetHandler): An instance of HeterogeneousGraphDatasetHandler managing the dataset.
            __graph_dataset (HeteroData): The heterogeneous graph dataset.
            __gnn (GNNEncoderInterface): The GNN encoder to be used in the edge regression model.
            _model (Model): The edge regression model.
            __train_data, __val_data, __test_data (HeteroData): Datasets for training, validation, and testing.
            __device (torch.device): The device on which to perform computations (GPU if available, else CPU).
            __trained_models_path (str): The path to save the trained GNN model.
            __tensorboard_path (str): The path for storing TensorBoard logs.
    """

    def __init__(self, graph_dataset_handler: HeterogeneousGraphDatasetHandler, gnn_encoder: GNNEncoderInterface):
        super().__init__()

        # Get the heterogeneous graph dataset
        self.__gdh = graph_dataset_handler
        self.__graph_dataset = graph_dataset_handler.get_graph_dataset()

        # Set the GNN encoder for the edge regression model
        self.__gnn = gnn_encoder

        # Initialize the edge regression model
        self._model = None

        # Split the dataset in training, validation and test sets
        self.__train_data, self.__val_data, self.__test_data = self.__edge_level_dataset_split()

        # Enable GPU computation (if available)
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model storage settings
        self.__trained_models_path = "..\\Src\\trained_models"
        self.__tensorboard_path = os.path.join(self.__trained_models_path, "runs")

    @overrides
    def train(self, num_epochs=300, lr=0.01):
        """
            Trains the GNN-based collaborative filtering model on the training set obtained by a partition of the
            heterogeneous graph dataset.

            This function supports TensorBoard to get the final plot of the training.

            Parameters:
                num_epochs (int, optional): Number of training epochs. Default is 300.
                lr (float, optional): Learning rate for training. Default is 0.01.

            Returns:
                None
        """
        print(f"Device: '{self.__device}\n'")

        # Initialize the GNN model with the specific GNN to use for the encoder
        self._model = Model(dataset=self.__graph_dataset, gnn_encoder=self.__gnn)
        self._model = self._model.to(self.__device)

        # Create a mini-batch loader that generates sub-graphs that can be used as input to the GNN
        # train_loader = self.__training_minibatch_loader()
        # validation_loader = self.__validation_minibatch_loader()

        # Define the optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        # Define a single training step
        def train_step(train_data):
            # Move data on device
            self._model.train()

            # Estimate the predicted rating
            optimizer.zero_grad()
            pred = self._model(
                train_data.x_dict,
                train_data.edge_index_dict,
                train_data['user', 'movie'].edge_label_index
            )

            # Extract the ground truth rating
            target = train_data['user', 'movie'].edge_label

            # Compute the MSE loss function
            loss = F.mse_loss(pred, target)

            # Compute the gradients of the model and update model weights
            loss.backward()
            optimizer.step()

            return float(loss)

        # Define a single evaluation step
        @torch.no_grad()
        def evaluation_step(data):
            # Move data on device
            data = data.to(self.__device)

            # Estimate the predicted rating
            self._model.eval()
            pred = self._model(
                data.x_dict,
                data.edge_index_dict,
                data['user', 'movie'].edge_label_index
            )
            pred = pred.clamp(min=0, max=5)

            # Extract the ground truth rating
            target = data['user', 'movie'].edge_label.float()

            # Compute the performance measures: RMSE and MAE
            rmse = F.mse_loss(pred, target).sqrt()
            mae = F.l1_loss(pred, target)

            return float(rmse), float(mae)

        # Store tensorboard information to get the plot at the end of the training
        tensorboard_foldername = os.path.join(self.__tensorboard_path, "Training_plot_" + self.__gnn.model_name + "_based_model_" + str(num_epochs) + "_epochs")

        if os.path.exists(tensorboard_foldername):    # If the folder already exists delete it
            shutil.rmtree(tensorboard_foldername)

        writer = SummaryWriter(tensorboard_foldername)

        # Train the GNN model
        for epoch in range(1, num_epochs):
            # Move data on device
            self.__train_data = self.__train_data.to(self.__device)

            # GNN model training step
            loss = train_step(self.__train_data)

            # GNN model evaluation step
            train_rmse, train_mae = evaluation_step(self.__train_data)
            val_rmse, val_mae = evaluation_step(self.__val_data)

            # Print training information
            print(f'Epoch: {epoch:03d}, Train loss: {loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}')

            # Add RMSE and MAE metrics to tensorboard (two separated plots)
            writer.add_scalar('RMSE/train', train_rmse, epoch)
            writer.add_scalar('RMSE/val', val_rmse, epoch)
            writer.add_scalar('MAE/train', train_mae, epoch)
            writer.add_scalar('MAE/val', val_mae, epoch)

        print(f'\nTraining TensorBoard information of model {self.__gnn.model_name} saved')

        # Store the trained model
        model_filename = os.path.join(self.__trained_models_path , self.__gnn.model_name + "_based_model_.pkl")
        torch.save(self._model, model_filename)
        print(f'\nTrained model {self.__gnn.model_name} saved')

    @overrides
    def evaluate_performance(self):
        """
            Evaluates the model on unseen data from the test set obtained by a partition of the heterogeneous graph
            dataset.

            Returns:
                None
        """
        print(f"Device: '{self.__device}\n'")

        with torch.no_grad():
            # Move data on device
            self.__test_data = self.__test_data.to(self.__device)

            # Estimate the predicted rating
            pred = self._model(
                self.__test_data.x_dict,
                self.__test_data.edge_index_dict,
                self.__test_data['user', 'movie'].edge_label_index
            )
            pred = pred.clamp(min=0, max=5)

            # Extract the ground truth rating
            target = self.__test_data['user', 'movie'].edge_label.float()

            # Compute the performance measures: RMSE and MAE
            rmse = F.mse_loss(pred, target).sqrt()
            mae = F.l1_loss(pred, target)
            print(f'Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}\n')

        # Prints a comparison between ground truth ratings and predicted rating for all data in the test set
        user_id = self.__test_data['user', 'movie'].edge_label_index[0].cpu().numpy()
        movie_id = self.__test_data['user', 'movie'].edge_label_index[1].cpu().numpy()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        print(pd.DataFrame({'userId': user_id, 'movieId': movie_id, 'pred_rating': pred, 'gt_rating': target}))

    @overrides
    def predict(self, user_id, movie_id, gt_rating = None):
        """
            Predict the rating for a given user and movie.

            Parameters:
                user_id (int): The ID of the user.
                movie_id (int): The ID of the movie.
                gt_rating (float, optional): The user rating for the movie. Defaults is None.

            Returns:
                Tuple containing the predicted rating and the ground truth rating.
        """
        # Map user and movie IDs (of the tabular dataset) to their corresponding indices in the graph dataset
        mapped_user_id = self.__gdh.unique_users_ids[self.__gdh.unique_users_ids['userId'] == user_id]['mappedUserId'].item()
        mapped_movie_id = self.__gdh.unique_movies_ids[self.__gdh.unique_movies_ids['movieId'] == movie_id][
            'mappedMovieId'].item()

        # Create edge_label_index for the given user and movie
        edge_label_index = (
            torch.tensor([mapped_user_id], dtype=torch.long),
            torch.tensor([mapped_movie_id], dtype=torch.long)
        )

        # Move the input data to the device
        data = self.__graph_dataset.to(self.__device)

        # Use the trained model for prediction
        with torch.no_grad():
            pred = self._model(
                data.x_dict,
                data.edge_index_dict,
                edge_label_index
            )
            pred = pred.clamp(min=0, max=5)

            # Extract the ground truth rating from the edge
            #target = data['user', 'rating', 'movie'].edge_label[edge_label_index[0][0]].item()

            # Extract the ground truth rating directly from the tabular dataset
            target_df = self.__gdh.users_ratings_df[
                (self.__gdh.users_ratings_df['userId'] == user_id) &
                (self.__gdh.users_ratings_df['movieId'] == movie_id)
            ]['rating']

            # Check if there are any matching items
            if not target_df.empty:
                target = target_df.iloc[0]
            else:
                target = np.nan

        return pred.item(), target

    def suggest_new_movie(self, user_id):
        """
            Suggests a new movie for a given user, based on the history of his past ratings.

            Parameters:
                user_id (int): The ID of the user.

            Returns:
                Tuple containing the suggested movie and the predicted rating.
        """
        # Upload some (updated) useful data from the graph dataset
        users_ratings_df = self.__gdh.users_ratings_df
        movies_df = self.__gdh.movies_df.rename(columns={'id': 'movieId'}, inplace=False)
        unique_users_ids = self.__gdh.unique_users_ids
        unique_movies_ids = self.__gdh.unique_movies_ids

        # Extract the mapped user id value
        mapped_user_id = unique_users_ids[unique_users_ids['userId'] == user_id]['mappedUserId'].values[0]

        # Select movies that user has not seen before
        movies_rated = users_ratings_df[users_ratings_df['mappedUserId'] == mapped_user_id]
        movies_not_rated = movies_df[~movies_df.index.isin(movies_rated['movieId'])]
        movies_not_rated = movies_not_rated.merge(unique_movies_ids, on='movieId')

        # Select a sample movie from the set of 'movies_not_rated'
        movie = movies_not_rated.sample(1)

        # Create new `edge_label_index` between the user and the movie
        edge_label_index = torch.tensor([
            mapped_user_id,
            movie.mappedMovieId.item()
        ])

        # Predict the user_id's rating for that movie
        with torch.no_grad():
            self.__test_data.to(self.__device)
            pred = self._model(
                self.__test_data.x_dict,
                self.__test_data.edge_index_dict,
                edge_label_index
            )
            pred = pred.clamp(min=0, max=5).detach().cpu().numpy()

        return movie, pred.item()

    def __edge_level_dataset_split(
            self,
            val_ratio=0.1,
            test_ratio=0.1,
            neg_sampling_ratio=0.0
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
        """
            Splits the edges of the graph dataset into training, validation, and test sets.

            Parameters:
                val_ratio (float): The portion of the dataset to use for validation.
                test_ratio (float): The portion of the dataset to use for test.
                neg_sampling_ratio (float): The ratio of negative edges generated for the evaluation.

            Returns:
                Tuple of training set, validation set, and test set.
        """
        # Split the set of edges into training (80%), validation (10%), and testing edges (10%).
        # Across the training edges, use 70% of edges for message passing, and 30% of edges for supervision.
        # Generate fixed negative edges for evaluation with a ratio of 2:1.
        # Negative edges will be generated on-the-fly during training.
        transform = T.RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            #disjoint_train_ratio=disjoint_train_ratio,
            neg_sampling_ratio=neg_sampling_ratio,
            #add_negative_train_samples=False,
            edge_types=('user', 'rating', 'movie'),
            rev_edge_types=('movie', 'rev_rating', 'user'),
        )
        train_data, val_data, test_data = transform(self.__graph_dataset)

        return train_data, val_data, test_data

    # Deprecated function: does not work on GPU
    def __training_minibatch_loader(
            self,
            batch_size=128,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0
    ) -> LinkNeighborLoader:
        """
            Creates a mini-batch loader that generates sub-graphs for training.

            Note: while this step is not strictly necessary for small-scale graphs, it is absolutely necessary to apply
            GNNs on larger graphs that do not fit onto GPU memory otherwise. To do this it samples multiple hops from
            both ends of an edge and creates a subgraph from it (using the loader.LinkNeighborLoader).

            Parameters:
                batch_size (int): The batch size.
                num_neighbors (list of int): The maximum number of neighbors to sample at each hop.
                neg_sampling_ratio (float): The ratio of negative edges to sample during training.

            Returns:
                LinkNeighborLoader: The train loader.
        """
        # Define seed edges
        edge_label_index = self.__train_data['user', 'rating', 'movie'].edge_label_index
        edge_label = self.__train_data['user', 'rating', 'movie'].edge_label

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        # First hop: sample at most 20 neighbors.
        # Second hop: sample at most 10 neighbors.
        # Negative edges will be sampled on-the-fly during training, with a ratio of 2:1.
        train_loader = LinkNeighborLoader(
            data=self.__train_data,
            num_neighbors=num_neighbors,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_label_index=(('user', 'rating', 'movie'), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )

        return train_loader

    # Deprecated function: does not work on GPU
    def __validation_minibatch_loader(
            self,
            batch_size=3 * 128,
            num_neighbors=[20, 10],
    ) -> LinkNeighborLoader:
        """
            Creates a mini-batch loader that generates sub-graphs for validation.

            Note: while this step is not strictly necessary for small-scale graphs, it is absolutely necessary to apply
            GNNs on larger graphs that do not fit onto GPU memory otherwise. To do this it samples multiple hops from
            both ends of an edge and creates a subgraph from it (using the loader.LinkNeighborLoader).

            Parameters:
                batch_size (int): The batch size.
                num_neighbors (list of int): The maximum number of neighbors to sample at each hop.

            Returns:
                LinkNeighborLoader: The validation loader.
        """
        # Define seed edges
        edge_label_index = self.__val_data["user", "rating", "movie"].edge_label_index
        edge_label = self.__val_data["user", "rating", "movie"].edge_label

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_label = torch.tensor(edge_label, dtype=torch.float)

        # First hop: sample at most 20 neighbors.
        # Second hop: sample at most 10 neighbors.
        val_loader = LinkNeighborLoader(
            data=self.__val_data,
            num_neighbors=num_neighbors,
            edge_label_index=(("user", "rating", "movie"), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=False,
        )

        return val_loader