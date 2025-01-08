import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # Localization network to predict affine transformation
        self.localization = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)  # 6 parameters for a 2D affine transformation
        )
        
        # Apply Kaiming initialization
        self._initialize_weights()
        
    # kaiming initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def knn_sampler(self, positions, properties, new_positions, k=3):
        """
        k-NN interpolation using torch to sample new properties at given new positions.

        Args:
            positions: Tensor of shape [num_points, 2], original 2D positions.
            properties: Tensor of shape [num_points, 3], original 3D properties.
            new_positions: Tensor of shape [num_new_points, 2], new 2D positions.
            k: Number of nearest neighbors to use for interpolation.

        Returns:
            Tensor of shape [num_new_points, 3], interpolated properties at new positions.
        """
        # Calculate pairwise distances between new_positions and positions
        distances = torch.cdist(new_positions, positions)  # Shape: [num_new_points, num_points]

        # Find the k nearest neighbors for each new_position
        knn_distances, knn_indices = torch.topk(distances, k=k, largest=False)

        # Gather the properties of the k nearest neighbors
        knn_properties = properties[knn_indices]  # Shape: [num_new_points, k, 3]

        # Compute the weights for each neighbor (inverse distance weighting)
        weights = 1 / (knn_distances + 1e-8)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights

        # Perform weighted sum of the properties
        new_properties = torch.sum(weights.unsqueeze(-1) * knn_properties, dim=1)

        return new_properties
        
                     
    def forward(self, positions, properties):
        '''
            positions: [N, 2] coordinates
            properties: [N, 3]
        '''

        # Flatten the positions for the localization network
        batch_size, _ = positions.size()

        # Predict the affine transformation matrix
        theta = self.localization(positions)
        theta = theta.view(batch_size, 2, 3)

        # Apply the affine transformation to (x, y) positions
        positions_homogeneous = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=1)  # Add ones for affine transform
        transformed_positions = torch.bmm(theta.view(-1, 2, 3), positions_homogeneous.view(-1, 3).unsqueeze(2)).squeeze(2)

        # sample properties with new positions
        output = self.knn_sampler(positions, properties, transformed_positions, k=5)
        
        return output


# ✅ **Test the Spatial Transformer**
if __name__ == "__main__":
    # Example input: [batch_size, num_points, 5] -> (x, y, r, g, b)
    input_tensor = torch.randn(100, 5)  # Batch of 1, 100 points, 5-dimensional input

    # Instantiate the model
    stn = SpatialTransformer()

    # Pass the input through the Spatial Transformer
    output_tensor = stn(input_tensor)

    print("Input shape:", input_tensor.shape)    # [1, 100, 5]
    print("Output shape:", output_tensor.shape)  # [1, 100, 5]

