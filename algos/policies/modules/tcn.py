import torch
from torch import nn


class TCNBlock(nn.Module):
    """TCN residual block with dilated convolutions."""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()

        # For causal convolution: padding = (kernel_size - 1) * dilation
        # This ensures output length equals input length
        padding = (kernel_size - 1) * dilation

        # First dilated convolution layer
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution layer
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Init conv weights
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.conv2 = nn.utils.weight_norm(self.conv2)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        skip = x

        out = self.conv1(x)
        # Trim to maintain causality (remove excessive padding)
        out = out[:, :, : x.size(2)]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]  # Trim
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            skip = self.downsample(skip)

        return self.relu(out + skip)


class TCN(nn.Module):
    """
    TCN baseline that processes raw RGB images and scalar states.
    Uses dilated convolutions for temporal modeling instead of attention.
    """

    def __init__(self, image_shape=(1, 3, 94, 94), state_dim=2, dim_model=512, tcn_channels=[1024, 768, 512, 512], kernel_size=3, dropout=0.1, final_ffn=False):
        super().__init__()

        self.dim_model = dim_model
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.final_ffn = final_ffn if tcn_channels[-1] == dim_model else True

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

        # TCN layers for temporal modeling
        layers = []
        num_levels = len(tcn_channels)
        input_channels = dim_model * 2  # Combined image + state features

        for i in range(num_levels):
            dilation_size = 2**i
            out_channels = tcn_channels[i]
            layers.append(TCNBlock(input_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout))
            input_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        self.output_head = nn.Sequential(
            nn.Linear(tcn_channels[-1], dim_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_model, dim_model)
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

        # Combine features and transpose as TCN expects (batch, channels, sequence_length)
        tcn_input = torch.cat([image_features, state_features], dim=-1).transpose(1, 2)

        # Process through TCN
        tcn_out = self.tcn(tcn_input).transpose(1, 2)  # (batch, time_seq, channels)

        # Generate output
        if self.final_ffn:
            out = self.layer_norm(self.output_head(tcn_out))
        else:
            out = self.layer_norm(tcn_out)

        return out
