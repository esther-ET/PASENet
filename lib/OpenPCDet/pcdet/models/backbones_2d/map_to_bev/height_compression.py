import torch.nn as nn
import torch.nn.functional as F


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']

        spatial_features = encoded_spconv_tensor.dense()
        # print("spatial_features_shape", spatial_features.shape)
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        # print("spatial_features_shape", spatial_features.shape)  # torch.Size([4, 256, 204, 180])

        batch_dict['spatial_features'] = F.interpolate(
            batch_dict['spatial_features'],
            size=(200, 176),
            mode='bilinear',
            align_corners=True
        )
        # print("spatial_features_shape after interpolation",batch_dict['spatial_features'].shape)  # torch.Size([4, 256, 200, 176])

        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        # print("batch_dict['spatial_features_stride']", batch_dict['spatial_features_stride'])
        return batch_dict
