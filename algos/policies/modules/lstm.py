import torch
from torch import nn


class LSTM(nn.Module):
    """
    LSTM baseline that processes raw RGB images and scalar states.
    Comparable to your transformer but uses LSTM for temporal modeling.
    """

    def __init__(self, image_shape=(1, 3, 94, 94), state_dim=2, dim_model=512, lstm_hidden=512, n_lstm_layers=4, dropout=0.1, final_ffn=False):
        super().__init__()

        self.dim_model = dim_model
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.final_ffn = final_ffn

        # CNN for image processing
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),  # 94x94 -> 47x47
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),  # 47x47 -> 24x24
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),  # 24x24 -> 12x12
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),  # 12x12 -> 6x6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),  # 6x6 -> 6x6
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, dim_model), nn.ReLU(), nn.Dropout(dropout)  # Global average pooling -> single feature vector
        )
        self.camera_projection = nn.Linear(dim_model * self.image_shape[0], dim_model)

        # MLP for state processing
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, dim_model), nn.ReLU(), nn.Dropout(dropout)
        )

        # LSTM for temporal modeling
        # Input: concatenated image + state features
        self.lstm = nn.LSTM(
            input_size=dim_model * 2,  # image_features + state_features
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, dim_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_model, dim_model)
        )
        # we skip the output head for now, not enforcing tcn_channels[-1]==dim_model
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, images, states):
        """
        Args:
            images: (batch, time_seq, num_cameras, C, H, W) or (batch, time_seq, C, H, W)
            states: (batch, time_seq, state_dim)
        Returns:
            output: (batch, time_seq, dim_model)
        """
        bs, time_seq = images.shape[:2]

        # Handle single camera case
        if len(images.shape) == 5:  # (batch, time_seq, C, H, W)
            images = images.unsqueeze(2)  # Add camera dimension

        # Encode images
        image_features = self.image_encoder(images.view(-1, *images.shape[3:]))  # (batch*time*cameras, dim_model)
        image_features = image_features.view(bs, time_seq, -1)
        image_features = self.camera_projection(image_features)  # (batch, time_seq, dim_model)

        # Encode states
        state_features = self.state_encoder(states.view(-1, states.shape[-1]))  # (batch*time, dim_model)
        state_features = state_features.view(bs, time_seq, -1)

        # Process through LSTM
        lstm_out, _ = self.lstm(torch.cat([image_features, state_features], dim=-1))

        # Generate output
        if self.final_ffn:
            out = self.layer_norm(self.output_head(lstm_out))
        else:
            out = self.layer_norm(lstm_out)

        return out
