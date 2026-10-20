# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see LICENSE file)
#
# Author: Fan Yang (fanyang1@ethz.ch)
# Robotic Systems Lab, ETH Zurich
# 2025

import torch
import torch.nn.functional as F

class DepthNoise(torch.nn.Module):
    def __init__(self,
        focal_length,
        baseline,
        min_depth,
        max_depth,
        filter_size=3,
        inlier_thred_range=(0.01, 0.05),
        prob_range=(0.4, 0.6),
        invalid_disp=1e7
    ):
        """
        A Simply PyTorch module to add realistic noise to depth images.

        Args:
            focal_length (float): Focal length of the camera (in pixels).
            baseline (float): Baseline distance between stereo cameras (in meters).
            min_depth (float): Minimum depth value after clamping.
            max_depth (float): Maximum depth value after clamping.
            filter_size (int): Kernel size for local mean disparity computation. (tuning based on image resolution)
            inlier_thred_range (tuple): Threshold range for normalized disparity differences.
            prob_range (tuple): Probability range for matching pixels.
            invalid_disp (float): Invalid disparity

        """
        super().__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.invalid_disp = invalid_disp
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.filter_size = filter_size
        
        weights, substitutes = self._compute_weights(filter_size)
        self.register_buffer('weights', weights.view(1, 1, filter_size, filter_size))
        self.register_buffer('substitutes', substitutes.view(1, 1, filter_size, filter_size))
        
        
    def _compute_weights(self, filter_size):
        """
        Compute weights and substitutes for disparity filtering.

        Args:
            filter_size (int): Kernel size for local mean disparity computation.
        """
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing='ij')
        sqr_radius = x_filter ** 2 + y_filter ** 2
        sqrt_radius = torch.sqrt(sqr_radius)
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()
        
        return weights, substitutes

    def filter_disparity(self, disparity):
        """
        Filter the disparity map using local mean disparity.
        
        Args:
            disparity (torch.Tensor): Input disparity map tensor of shape (B, C, H, W).
        """
        B, _, H, W = disparity.shape
        device = disparity.device
        center = self.filter_size // 2

        output_disparity = torch.full_like(disparity, self.invalid_disp)

        prob = torch.rand(B, 1, 1, 1, device=device) * (self.prob_range[1] - self.prob_range[0]) + self.prob_range[0]
        random_mask = (torch.rand(B, 1, H, W, device=device) < prob)

        # Compute mean disparity
        weighted_disparity = F.conv2d(disparity, self.weights, padding=center)

        # Compute differences
        differences = torch.abs(disparity - weighted_disparity)

        # Normalize differences based on current image statistics for consistent thresholding
        differences_flat = differences.view(B, -1)  # Flatten spatial dimensions
        mean_diff = torch.mean(differences_flat, dim=1, keepdim=True)
        std_diff = torch.std(differences_flat, dim=1, keepdim=True) + 1e-6  # Add epsilon to avoid division by zero

        # Normalize differences: (diff - mean) / std, then shift to [0, 1] range approximately
        normalized_differences_flat = (differences_flat - mean_diff) / std_diff
        normalized_differences = normalized_differences_flat.view_as(differences)

        # Use parameter-based threshold on normalized differences
        threshold = torch.rand(B, 1, 1, 1, device=device) * (self.inlier_thred_range[1] - self.inlier_thred_range[0]) + self.inlier_thred_range[0]
        update_mask = (normalized_differences < threshold) & random_mask & (disparity > 0)

        # Compute output value: round with 1/32 precision
        disparity = torch.round(disparity * 32.0) / 32.0

        # Update output disparity
        output_disparity = torch.where(update_mask, disparity, output_disparity)

        # Apply substitutes to fill neighboring pixels
        filled_values = F.conv2d(update_mask.float() * disparity, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        average_filled_values = filled_values / counts
        output_disparity = torch.where(counts >= 1, average_filled_values, output_disparity)
        output_disparity = torch.round((output_disparity + 0.1 * torch.randn_like(output_disparity)) * 32.0) / 32.0

        return output_disparity

    def forward(self, depth) -> torch.Tensor:
        # correct input shape
        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1) # add channel dimension
        
        # check dimension (B, 1, H, W)
        assert depth.shape[1] == 1, "Input depth tensor must have shape (B, 1, H, W)."
        assert len(depth.shape) == 4, "Input depth tensor must have shape (B, 1, H, W)."
        
        # Mask invalid or out-of-range depth values
        valid_mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        depth = torch.where(valid_mask, depth, self.max_depth)
    
        # Step 1: Convert depth to disparity
        disparity = (self.focal_length * self.baseline / depth) * valid_mask

        # Step 2: Filter the disparity map
        filtered_disparity = self.filter_disparity(disparity)

        # Step 3: Recompute depth from disparity
        depth = self.focal_length * self.baseline / filtered_disparity
            
        # Step 4: Set invalid depth values to 0.0 (values outside valid range are not measurable)
        # Values below min_depth are too close to measure reliably
        # Values above max_depth are too far to measure reliably
        depth[depth > self.max_depth] = 0.0
        depth[depth < self.min_depth] = 0.0
        depth[~valid_mask] = 0.0

        return depth
    
# Example usage
if __name__ == "__main__":
    import cv2
    import yaml
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from dataloader import DepthImageDataset
    from torch.utils.data import DataLoader
    
    from network.image_utils import RandomCropDownsample
    
    def normalize_depth(depth, min_depth, max_depth, is_log):
        depth = torch.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
        depth = torch.clamp(depth, min_depth, max_depth) # Clamp the depth values
        depth = torch.log(depth + 1.0) if is_log else depth
        return depth
    
    
    config_path = '/home/fanyang/Projets/world_model_pretrain/config/pretrain.yaml'
    
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    config = load_config(config_path)
    dataloader_config = config.get('dataloader', {})
    training_config = config.get('training', {})
    
    min_depth = dataloader_config.get('min_depth', 0.1)
    max_depth = dataloader_config.get('max_depth', 10.0)
    focal_length = dataloader_config.get('focal_length', 50.0)
    baseline = dataloader_config.get('baseline', 0.12)
    
    # Detect whether or not is at cluster
    is_cluster = False
    if 'cluster' in os.getcwd():
        print("Running on cluster.")
        is_cluster = True
    else:
        print("Running on local machine.")

    depth_dataset = DepthImageDataset(dataloader_config, is_cluster)
    data_loader = DataLoader(depth_dataset, batch_size=dataloader_config.get('batch_size', 256), shuffle=dataloader_config.get('shuffle', True), num_workers=dataloader_config.get('num_workers', 4))

    # Get a single batch from the dataloader
    depth_batch = next(iter(data_loader))
    depth = depth_batch
    batch_size = depth.shape[0]
    
    normalize_depth_fn = lambda x: normalize_depth(x, min_depth, max_depth, is_log=training_config['log_space'])
    
    depth_noise = DepthNoise(focal_length=focal_length,
                             baseline=baseline, 
                             min_depth=min_depth,
                             max_depth=max_depth)
    depth_noise = torch.jit.script(depth_noise)
    
    # Normalize the depth values
    depth = normalize_depth_fn(depth)

    # Add noise to the depth images
    noisy_depth = depth_noise(depth)
    
    # visualization resolution x4 of the original depth image
    H_orig, W_orig = depth.shape[1], depth.shape[2]
    H_vis, W_vis = H_orig * 5, W_orig * 5
    
    for i in range(batch_size):
        clip_depth = depth[i].clip(0.1, 10.0)
        disp_depth = clip_depth.squeeze().detach().cpu().numpy()
        disp_nosie_depth = noisy_depth[i].squeeze().detach().cpu().numpy()
        # enlarge the resolution for better visualization
        disp_depth = cv2.resize(disp_depth, (W_vis, H_vis))
        disp_nosie_depth = cv2.resize(disp_nosie_depth, (W_vis, H_vis))
        # normalize the depth values to 0-1
        disp_depth = (disp_depth - disp_depth.min()) / (disp_depth.max() - disp_depth.min() + 1e-7)
        disp_nosie_depth = (disp_nosie_depth - disp_nosie_depth.min()) / (disp_nosie_depth.max() - disp_nosie_depth.min())
        
        # Merge the images horizontally
        merged_image = cv2.hconcat([disp_depth, disp_nosie_depth])
        
        # Display the merged image
        cv2.imshow("Depth and Noisy Depth Image", merged_image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break
        cv2.destroyAllWindows()
    
    print("Noisy depth shape:", noisy_depth.shape)

