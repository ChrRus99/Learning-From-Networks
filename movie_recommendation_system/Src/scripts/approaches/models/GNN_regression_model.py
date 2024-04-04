# Util
from abc import ABC

# Pytorch geometric
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import Linear, GraphConv, SAGEConv, GATConv, to_hetero

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
        - Colab Notebooks and Video Tutorials:
            https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
        - Source code for torch_geometric.nn.models.autoencoder:
            https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/autoencoder.html
        - PyTorch Geometric tutorial: Graph Autoencoders & Variational Graph Autoencoders:
            https://www.youtube.com/watch?v=qA6U4nIK62E&ab_channel=AntonioLonga
"""


class GNNEncoderInterface(torch.nn.Module, ABC):
    """
        Graph Neural Network (GNN) Encoder Interface.

        This interface defines a GNN model for learning enriched node representations from the surrounding sub-graphs,
        which can be then used to derive edge-level predictions.

        For defining our heterogeneous GNN, it makes use of PyTorch Geometric's nn.<TypeConv> layers and the
        nn.to_hetero() function. The to_hetero() function transforms a GNN defined on homogeneous graphs to be applied
        on heterogeneous ones.

        Args:
            hidden_channels (int): Number of hidden channels in the GCN layers.
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels


class GCNEncoder(GNNEncoderInterface):
    """
        Graph Convolutional Network (GCN) Encoder.

        This class defines a GCN Encoder with two message passing layers for the encoding.

        Args:
            hidden_channels (int): Number of hidden channels in the GCN layers.
            out_channels (int): Number of output channels in the GCN layers.

        Attributes:
            conv1 (torch_geometric.nn.conv.GraphConv): First GCN layer.
            conv2 (torch_geometric.nn.conv.GraphConv): Second GCN layer.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__(hidden_channels)
        self.conv1 = GraphConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GraphConv((-1, -1), out_channels, add_self_loops=False)
        self.model_name = 'GCN'

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
            Forward pass of the GCN Encoder.

            Args:
                x (torch.Tensor): Edge features.
                edge_index (Tensor): Edge index in x.

            Returns:
                torch.Tensor: Encoded graph features.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEEncoder(GNNEncoderInterface):
    """
        GraphSAGE Network Encoder.

        This class defines a GraphSAGE Network Encoder with two message passing layers for the encoding.

        Args:
            hidden_channels (int): Number of hidden channels in the SAGEConv layers.
            out_channels (int): Number of output channels in the SAGEConv layers.

        Attributes:
            conv1 (torch_geometric.nn.conv.GraphConv): First SAGEConv layer.
            conv2 (torch_geometric.nn.conv.GraphConv): Second SAGEConv layer.
    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__(hidden_channels)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.model_name = 'GraphSAGE'

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
            Forward pass of the GraphSAGE Encoder.

            Args:
                x (torch.Tensor): Edge features.
                edge_index (torch.Tensor): Edge index in x.

            Returns:
                torch.Tensor: Encoded graph features.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GATEncoder(GNNEncoderInterface):
    """
        Graph Attention Network (GAT) Encoder.

        This class defines a GAT Encoder with two message passing layers for the encoding.

        Args:
            hidden_channels (int): Number of hidden channels in the GAT layers.
            out_channels (int): Number of output channels in the GAT layers.

        Attributes:
            conv1 (torch_geometric.nn.conv.GATConv): First GAT layer.
            lin1 (torch.nn.Linear): First linear layer.
            conv2 (torch_geometric.nn.conv.GATConv): Second GAT layer.
            lin2 (torch.nn.Linear): Second linear layer.

    """

    def __init__(self, hidden_channels, out_channels):
        super().__init__(hidden_channels)
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)
        self.model_name = 'GAT'

    def forward(self, x, edge_index) -> Tensor:
        """
            Forward pass of the GAT Encoder.

            Args:
                x (torch.Tensor): Edge features.
                edge_index (torch.Tensor): Edge index in x.

            Returns:
                torch.Tensor: Encoded graph features.
        """
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


class EdgeDecoder(torch.nn.Module):
    """
        Edge Regression Decoder.

        This class defines an Edge Regression Decoder for predicting the rating for the encoded user-movie combination.

        Args:
            hidden_channels (int): Number of hidden channels.

        Attributes:
            lin1 (torch.nn.Linear): First linear layer.
            lin2 (torch.nn.Linear): Second linear layer.
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index) -> Tensor:
        """
            Forward pass of the Edge Regression Decoder.

            Args:
                z_dict (dict): Dictionary containing edge embeddings.
                edge_label_index (tuple): Tuple containing row and column index of the edge to do regression about.

            Returns:
                torch.Tensor: Predicted edge rating.
        """
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    """
        Graph Auto-Encoder (GAE) Edge Regression Model.

        This class defines a Graph Auto-Encoder (GAE) edge regression model for the MovieLens Heterogeneous Graph
        Dataset.

        Args:
            dataset (your_dataset): The MovieLens Heterogeneous Graph Dataset.
            gnn_encoder (GNNEncoderInterface): The specific type of GNN encoder for learning graph representations.

        Attributes:
            encoder (torch_geometric.nn.conv.to_hetero): Heterogeneous GNN encoder.
            decoder (EdgeDecoder): Edge Regression Decoder.
    """

    def __init__(self, dataset, gnn_encoder: GNNEncoderInterface):
        super().__init__()
        self.encoder = to_hetero(gnn_encoder, dataset.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(gnn_encoder.hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        """
            Forward pass of the GAE edge regression model.

            Args:
                x_dict (dict): Dictionary containing edge features.
                edge_index_dict (dict): The edge index in the x_dict dictionary.
                edge_label_index (tuple): Tuple containing row and column index of the edge in z_dict.

            Returns:
                torch.Tensor: Predicted edge rating.
        """
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def encode(self, x_dict, edge_index_dict):
        """
            Encodes input data using the GNN encoder.

            Args:
                x_dict (dict): Dictionary containing edge features.
                edge_index_dict (dict): The edge index in the x_dict dictionary.

            Returns:
                torch.Tensor: Encoded node representations.
        """
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, x_dict, edge_index_dict, edge_label_index):
        """
            Decodes node representations to obtain edge prediction.

            Args:
                x_dict (dict): Dictionary containing edge features.
                edge_index_dict (dict): The edge index in the x_dict dictionary.
                edge_label_index (tuple): Tuple containing row and column index of the edge in z_dict.

            Returns:
                torch.Tensor: Predicted edge rating.
        """
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)