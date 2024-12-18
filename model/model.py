import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        numeric_input_dim: int,
        cat_input_dim: int,
        num_districts: int,
        district_emb_dim: int,
        num_properties: int,
        property_emb_dim: int,
        out_features: int = 1,
        dropout_rate: float = 0.15,
    ):
        """
        Initializes the neural network model, including embedding layers for district and property type.
        """
        super().__init__()
        self.district_embedding = nn.Embedding(num_districts, district_emb_dim)
        self.property_embedding = nn.Embedding(num_properties, property_emb_dim)

        total_input_dim = (
            numeric_input_dim + cat_input_dim + district_emb_dim + property_emb_dim
        )
        self.model = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, out_features),
        )

    def forward(
        self,
        numeric_x: torch.Tensor,
        cat_x: torch.Tensor,
        district_ids: torch.Tensor,
        property_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        dist_emb = self.district_embedding(district_ids)
        prop_emb = self.property_embedding(property_ids)
        combined = torch.cat([numeric_x, cat_x, dist_emb, prop_emb], dim=1)
        return self.model(combined)
