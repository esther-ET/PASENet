import numpy as np
import torch
import torch.nn as nn
from lib.pointpillars_with_TANet.torchplus.nn.modules.common import Sequential
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def visualize_tensor(tensor):
    """
    Visualize the spatial features.

    Args:
        spatial_features: torch.Tensor of shape (B, C, H, W)
    """
    # Viridis色带的颜色映射规则如下：
    #
    #     低值区域：色带的开始是深蓝色，表示数据中的低值。
    #     中值区域：随着数据值的增加，颜色逐渐过渡到绿色，然后是黄色，表示数据的中间范围。
    #     高值区域：色带的结束是白色，表示数据中的高值。
    B, C, H, W = tensor.shape

    num_features_to_visualize = 2  # 你可以根据需要调整这个数字

    fig, axs = plt.subplots(num_features_to_visualize, figsize=(180, 157))# HW
    axs = axs.flatten()  # 将axs转换为1D数组，方便索引

    for i in range(min(num_features_to_visualize, C)):
        # 选择第一个批次的特征图
        feature_map = tensor[0, i, :, :].cpu().detach().numpy()
        axs[i].imshow(feature_map, cmap='viridis')
        axs[i].set_title(f'Feature Map {i}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()


class SemanticSegmentationBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemanticSegmentationBranch, self).__init__()

        # 定义残差块
        self.residual_blocks = nn.Sequential(
            self._make_residual_block(64, 128),
            # self._make_residual_block(128, 256),

        )

        # 最大池化层
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 上采样层
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 卷积层
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)

        # BatchNorm 和 ReLU
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def _make_residual_block(self, in_channels, out_channels):
        # 残差块
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): #64--->64
        # 残差块
        x = self.residual_blocks(x)

        # 最大池化
        x = self.maxpool1(x)
        # x = self.maxpool2(x)

        # 上采样
        x = self.upsample1(x)
        # x = self.upsample2(x)

        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)  # jia
        x = F.relu(self.bn2(x))  # jia

        # 输出语义掩码

        return torch.sigmoid(x)  # 使用 sigmoid 激活函数生成概率图 (0,1)


class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()

    def forward(self, features, seg_mask):
        # 特征融合：根据语义分割掩码重新加权特征图
        return (1 + seg_mask) * features  #(1 + seg_mask) * features   # self.fusion(unet_output, seg_mask)


class SemanticContextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemanticContextEncoder, self).__init__()

        # 主检测分支（U-Net结构）
        # self.unet = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        # 语义分割分支
        self.segmentation_branch = SemanticSegmentationBranch(in_channels, out_channels)

        # 融合模块
        self.fusion = FusionModule()

    def forward(self, x):  # (64--->3)
        # 主检测分支
        # unet_output = self.unet(x)

        # 语义分割分支生成语义掩码
        seg_mask = self.segmentation_branch(x)  # (4,3,496,432)

        # 特征融合
        # fused_features = self.fusion(unet_output, seg_mask)
        # matched_seg_mask = self.seg_mask_expand(seg_mask)  # matched_seg_mask 的形状为 (B, 64, H, W)
        summed = torch.sum(seg_mask, dim=1, keepdim=True)
        normal_summed = torch.sigmoid(summed)
        matched_seg_mask = normal_summed.repeat(1, 64, 1, 1)
        fused_features = self.fusion(x, matched_seg_mask)
        # print("segmask:", torch.max(matched_seg_mask), torch.min(matched_seg_mask))
        return fused_features, seg_mask

class Segmask(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.num_class = num_class
        self.sce = SemanticContextEncoder(in_channels=64, out_channels=3)

        # 4,64,496,432


    def visualize_gt_boxes(self, spatial_features, gt_boxes, voxel_size, point_cloud_range,frame_ids):
        spatial_features = spatial_features.cpu().detach().numpy()
        gt_boxes = gt_boxes.cpu().detach().numpy()
        B, C, H, W = spatial_features.shape
        N = gt_boxes.shape[1]

        bev_boxes = []

        for batch_idx in range(B):
            batch_boxes = []
            for n in range(N):
                box = gt_boxes[batch_idx][n]
                gt_x, gt_y, gt_dx, gt_dy, heading = box[0], box[1], box[3], box[4], box[6]
                bbox_x = (gt_x - point_cloud_range[0]) / voxel_size[0]
                bbox_y = (gt_y - point_cloud_range[1]) / voxel_size[1]
                bbox_dx = gt_dx / voxel_size[0]
                bbox_dy = gt_dy / voxel_size[1]

                # Calculate the four corners of the bounding box
                corners = np.array([
                    [bbox_x - bbox_dx / 2, bbox_y - bbox_dy / 2],
                    [bbox_x + bbox_dx / 2, bbox_y - bbox_dy / 2],
                    [bbox_x + bbox_dx / 2, bbox_y + bbox_dy / 2],
                    [bbox_x - bbox_dx / 2, bbox_y + bbox_dy / 2]
                ])

                # Rotate the corners around the center
                rotation_matrix = np.array([
                    [np.cos(-heading), -np.sin(-heading)],
                    [np.sin(-heading), np.cos(-heading)]
                ])
                rotated_corners = np.dot(corners - [bbox_x, bbox_y], rotation_matrix) + [bbox_x, bbox_y]

                # Convert to integer indices
                rotated_corners = np.clip(rotated_corners, 0, [W - 1, H - 1]).astype(int)

                batch_boxes.append(rotated_corners.tolist())
            bev_boxes.append(batch_boxes)

            # Plotting the BEV boxes
        for batch_idx, batch_boxes in enumerate(bev_boxes):
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_title(f'Batch {batch_idx} - Frame ID: {frame_ids[batch_idx]}')
            ax.set_aspect('equal')

            # Visualize spatial features
            max_sum = -np.inf
            selected_channel = 0

            for c in range(C):
                channel_sum = spatial_features[batch_idx, c, :, :].sum().item()
                if channel_sum > max_sum:
                    max_sum = channel_sum
                    selected_channel = c

            feature_map = spatial_features[batch_idx, selected_channel, :, :]
            ax.imshow(feature_map, cmap='viridis') #viridis

            for box in batch_boxes:
                polygon = Polygon(box, closed=True, edgecolor='r', facecolor='none', linewidth=1)
                ax.add_patch(polygon)

            plt.gca().invert_yaxis()
            plt.show()

    # def match_box_to_bev(self, spatial_features, gt_boxes):
    #     """
    #     Args:
    #         spatial_features: torch.Tensor of shape (B, C, H, W)
    #         gt_boxes: torch.Tensor of shape (B, N, 8) [x, y, z, dx, dy, dz, heading, class]
    #     Returns:
    #         foreground_mask: torch.Tensor of shape (B, C, H, W) with 1 for foreground voxels, 0 for background
    #     """
    #     B, C, H, W = spatial_features.shape  # ([4, 64, 496, 432])
    #     N = gt_boxes.shape[1]
    #     voxel_size = self.voxel_size
    #     point_cloud_range = self.point_cloud_range  # [x_min, y_min, z_min, x_max, y_max, z_max]
    #
    #     # Initialize the foreground mask on the same device as spatial_features
    #     foreground_mask = torch.zeros((B, self.num_class, H, W), dtype=torch.int32, device=spatial_features.device)
    #
    #     # Convert voxel_size and point_cloud_range to tensors on the same device
    #     voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=spatial_features.device)
    #     point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32, device=spatial_features.device)
    #
    #     for batch_idx in range(B):
    #         batch_boxes = []
    #         for n in range(N):
    #             box = gt_boxes[batch_idx, n]  # Shape: [8]
    #             gt_x, gt_y, gt_dx, gt_dy, heading, box_class = box[0], box[1], box[3], box[4], box[6], box[7]
    #
    #             # Convert box coordinates to BEV coordinates
    #             bbox_x = (gt_x - point_cloud_range[0]) / voxel_size[0]
    #             bbox_y = (gt_y - point_cloud_range[1]) / voxel_size[1]
    #             bbox_dx = gt_dx / voxel_size[0]
    #             bbox_dy = gt_dy / voxel_size[1]
    #
    #             # Calculate the four corners of the bounding box
    #             corners = torch.tensor([
    #                 [bbox_x - bbox_dx / 2, bbox_y - bbox_dy / 2],
    #                 [bbox_x + bbox_dx / 2, bbox_y - bbox_dy / 2],
    #                 [bbox_x + bbox_dx / 2, bbox_y + bbox_dy / 2],
    #                 [bbox_x - bbox_dx / 2, bbox_y + bbox_dy / 2]
    #             ], device=spatial_features.device)
    #
    #             # Rotate the corners around the center
    #             rotation_matrix = torch.tensor([
    #                 [torch.cos(-heading), -torch.sin(-heading)],
    #                 [torch.sin(-heading), torch.cos(-heading)]
    #             ], device=spatial_features.device)
    #             center = torch.tensor([bbox_x, bbox_y], device=spatial_features.device)
    #             rotated_corners = torch.mm(corners - center, rotation_matrix.T) + center
    #
    #             # Clip the corners to the BEV grid boundaries
    #             min_bound = torch.tensor(0, device=spatial_features.device)  # 将 0 转换为张量
    #             max_bound = torch.tensor([W - 1, H - 1], device=spatial_features.device)
    #             rotated_corners = torch.clamp(rotated_corners, min=min_bound, max=max_bound)
    #
    #             # Convert to integer indices
    #             rotated_corners = rotated_corners.to(torch.int32)
    #
    #             # Check if any corner is out of bounds
    #             if torch.any(rotated_corners < 0) or torch.any(rotated_corners[:, 0] >= W) or torch.any(
    #                     rotated_corners[:, 1] >= H):
    #                 continue  # Skip this box if any corner is out of bounds
    #
    #             # Append the rotated corners and class to batch_boxes
    #             class_tensor = torch.tensor([box_class], device=spatial_features.device).expand(4, 1)  # Shape: [4, 1]
    #             batch_boxes.append(torch.cat([rotated_corners, class_tensor], dim=1))  # Shape: [4, 3]
    #
    #         if len(batch_boxes) == 0:
    #             continue  # Skip if no valid boxes in this batch
    #
    #         # Stack all boxes in the batch
    #         bev_boxes = torch.stack(batch_boxes)  # Shape: [N_boxes, 4, 3] (4 corners + 1 class)
    #
    #         # Process each box in the batch
    #         for box in bev_boxes:
    #             # Extract the x and y coordinates of the corners
    #             x_coords = box[:4, 0]  # 4 x coordinates
    #             y_coords = box[:4, 1]  # 4 y coordinates
    #
    #             # Find the minimum and maximum x and y coordinates
    #             min_x = int(torch.min(x_coords).item())  # 转换为整数
    #             max_x = int(torch.max(x_coords).item())  # 转换为整数
    #             min_y = int(torch.min(y_coords).item())  # 转换为整数
    #             max_y = int(torch.max(y_coords).item())  # 转换为整数
    #
    #             # Extract the class and convert to scalar
    #             box_class = int(box[0, 2].item())  # 将张量转换为整数标量
    #
    #             # Set the values within the rectangle to 1 for the corresponding class
    #             foreground_mask[batch_idx, box_class-1, min_y:max_y + 1, min_x:max_x + 1] = 1
    #
    #     return foreground_mask

    # CPU version
    # def match_box_to_bev(self, spatial_features, gt_boxes):
    #     """
    #     Args:
    #         spatial_features: torch.Tensor of shape (B, C, H, W)
    #         gt_boxes: torch.Tensor of shape (B, N, 8) [x, y, z, dx, dy, dz, heading,class]
    #         voxel_size: list or tuple of 3 elements [vx, vy, vz]
    #         point_cloud_range: list or tuple of 6 elements [x_min, y_min, z_min, x_max, y_max, z_max]
    #     Returns:
    #         foreground_mask: torch.Tensor of shape (B, C, H, W) with 1 for foreground voxels, 0 for background
    #     """
    #     B, C, H, W = spatial_features.shape  # ([4, 64, 496, 432])
    #     N = gt_boxes.shape[1]
    #     voxel_size = self.voxel_size
    #     # x_min, y_min, z_min, x_max, y_max, z_max
    #     point_cloud_range = self.point_cloud_range
    #
    #     # Initialize the foreground mask
    #     foreground_mask = torch.zeros((B, self.num_class, H, W), dtype=torch.int32, device=spatial_features.device)
    #
    #     for batch_idx in range(B):
    #         batch_boxes = []
    #         for n in range(N):
    #             box = gt_boxes[batch_idx][n].cpu().detach().numpy()
    #             gt_x, gt_y, gt_dx, gt_dy, heading = box[0], box[1], box[3], box[4], box[6]
    #             #  class
    #             box_class = box[7]
    #             # print(box_class) 1 2 3
    #
    #             bbox_x = (gt_x - point_cloud_range[0]) / voxel_size[0]
    #             bbox_y = (gt_y - point_cloud_range[1]) / voxel_size[1]
    #             bbox_dx = gt_dx / voxel_size[0]
    #             bbox_dy = gt_dy / voxel_size[1]
    #
    #             # Calculate the four corners of the bounding box
    #             corners = np.array([
    #                 [bbox_x - bbox_dx / 2, bbox_y - bbox_dy / 2],
    #                 [bbox_x + bbox_dx / 2, bbox_y - bbox_dy / 2],
    #                 [bbox_x + bbox_dx / 2, bbox_y + bbox_dy / 2],
    #                 [bbox_x - bbox_dx / 2, bbox_y + bbox_dy / 2]
    #             ])
    #
    #             # Rotate the corners around the center
    #             rotation_matrix = np.array([
    #                 [np.cos(-heading), -np.sin(-heading)],
    #                 [np.sin(-heading), np.cos(-heading)]
    #             ])
    #             rotated_corners = np.dot(corners - [bbox_x, bbox_y], rotation_matrix) + [bbox_x, bbox_y]
    #
    #             # Convert to integer indices
    #             rotated_corners = np.clip(rotated_corners, 0, [W - 1, H - 1]).astype(int)
    #
    #             # Check if any corner is out of bounds
    #             if np.any(rotated_corners < 0) or np.any(rotated_corners[:, 0] >= W) or np.any(
    #                     rotated_corners[:, 1] >= H):
    #                 continue  # Skip this box if any corner is out of bounds
    #
    #             box_class = [box_class.tolist()] * 2
    #             rotated_corners = rotated_corners.tolist()
    #             rotated_corners.append(box_class)
    #             batch_boxes.append(rotated_corners)  # [[0, 248], [0, 248], [0, 248], [0, 248], [0.0]]
    #
    #         # B,N(N boxes),4(4 corners),2(x,y)    --> B,N(N boxes),4(4 corners)+1(class),2(x,y)
    #         bev_boxes = torch.tensor(batch_boxes, dtype=torch.int32,
    #                                  device=spatial_features.device)  # torch.Size([4, 6, 4, 2]) torch.Size([4, 9, 4, 2])
    #         # print(bev_boxes.shape)
    #         # 获取四个corner的最小x，最大x，最小y，最大y
    #         # torch.Size([38, 5, 2])
    #         for box in bev_boxes:
    #             # Extract the x and y coordinates of the corners
    #             x_coords = box[:3, 0]  # 4 x and 1 class
    #
    #             y_coords = box[:3, 1]  # 4 y and 1 class
    #
    #             # Find the minimum and maximum x and y coordinates
    #             min_x = torch.min(x_coords)
    #             max_x = torch.max(x_coords)
    #             min_y = torch.min(y_coords)
    #             max_y = torch.max(y_coords)
    #             # print(f'min_x,max_x: {min_x},{max_x}')
    #             # print(f'min_y,max_y: {min_y},{max_y}')
    #
    #             # class
    #             box_class = box[4, 0]
    #             # print(box_class)
    #
    #             # Set the values within the rectangle to 1
    #             foreground_mask[batch_idx, :, min_y:max_y + 1, min_x:max_x + 1] = 0
    #             # class 1 2 3
    #             foreground_mask[batch_idx, box_class - 1, min_y:max_y + 1, min_x:max_x + 1] = 1
    #
    #     return foreground_mask
    def match_box_to_bev(self, spatial_features, gt_boxes, mask_type='voxel'):
        """
        优化后的函数，生成更精确的 BEV 语义掩码。

        Args:
            spatial_features: torch.Tensor of shape (B, C, H, W)，体素特征
            gt_boxes: torch.Tensor of shape (B, N, 8) [x, y, z, dx, dy, dz, heading, class]
            mask_type: str, 'voxel' 或 'box'，指定生成掩码的类型
        Returns:
            foreground_mask: torch.Tensor of shape (B, C, H, W) with 1 for foreground voxels, 0 for background
        """
        B, C, H, W = spatial_features.shape  # ([4, 64, 496, 432])
        N = gt_boxes.shape[1]
        voxel_size = self.voxel_size
        point_cloud_range = self.point_cloud_range  # [x_min, y_min, z_min, x_max, y_max, z_max]

        # Initialize the foreground mask
        foreground_mask = torch.zeros((B, self.num_class, H, W), dtype=torch.int32, device=spatial_features.device)

        # Convert voxel_size and point_cloud_range to tensors on the same device
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=spatial_features.device)
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32, device=spatial_features.device)

        for batch_idx in range(B):
            # 判断每个体素是否为空（全零体素为空）
            non_empty_mask = ~torch.all(spatial_features[batch_idx] == 0, dim=0)  # (H, W)

            for n in range(N):
                box = gt_boxes[batch_idx, n]  # Shape: [8]
                gt_x, gt_y, gt_dx, gt_dy, heading = box[0], box[1], box[3], box[4], box[6]
                box_class = int(box[7].item())  # 将 box_class 转换为整数

                # Convert box coordinates to BEV coordinates
                bbox_x = (gt_x - point_cloud_range[0]) / voxel_size[0]
                bbox_y = (gt_y - point_cloud_range[1]) / voxel_size[1]
                bbox_dx = gt_dx / voxel_size[0]
                bbox_dy = gt_dy / voxel_size[1]

                # Calculate the four corners of the bounding box
                corners = torch.tensor([
                    [bbox_x - bbox_dx / 2, bbox_y - bbox_dy / 2],
                    [bbox_x + bbox_dx / 2, bbox_y - bbox_dy / 2],
                    [bbox_x + bbox_dx / 2, bbox_y + bbox_dy / 2],
                    [bbox_x - bbox_dx / 2, bbox_y + bbox_dy / 2]
                ], device=spatial_features.device)

                # Rotate the corners around the center
                rotation_matrix = torch.tensor([
                    [torch.cos(-heading), -torch.sin(-heading)],
                    [torch.sin(-heading), torch.cos(-heading)]
                ], device=spatial_features.device)
                center = torch.tensor([bbox_x, bbox_y], device=spatial_features.device)
                rotated_corners = torch.mm(corners - center, rotation_matrix.T) + center

                # Clip the corners to the BEV grid boundaries
                min_bound = torch.tensor(0, device=spatial_features.device)  # 将 0 转换为张量
                max_bound = torch.tensor([W - 1, H - 1], device=spatial_features.device)
                rotated_corners = torch.clamp(rotated_corners, min=min_bound, max=max_bound)

                # Convert to integer indices
                rotated_corners = rotated_corners.to(torch.int32)

                # Check if any corner is out of bounds
                if torch.any(rotated_corners < 0) or torch.any(rotated_corners[:, 0] >= W) or torch.any(
                        rotated_corners[:, 1] >= H):
                    continue  # Skip this box if any corner is out of bounds

                # Generate mask based on mask_type
                if mask_type == 'voxel':
                    # Voxel-type mask: Only mark non-empty voxels that overlap with the box
                    min_x = torch.min(rotated_corners[:, 0])
                    max_x = torch.max(rotated_corners[:, 0])
                    min_y = torch.min(rotated_corners[:, 1])
                    max_y = torch.max(rotated_corners[:, 1])

                    # Create a binary mask for the box
                    box_mask = torch.zeros((H, W), dtype=torch.int32, device=spatial_features.device)
                    box_mask[min_y:max_y + 1, min_x:max_x + 1] = 1

                    # Only mark non-empty voxels within the box
                    box_mask = box_mask * non_empty_mask  # 只保留非空体素

                    # Mark only the non-empty voxels that overlap with the box
                    foreground_mask[batch_idx, box_class - 1] += box_mask
                    foreground_mask[batch_idx, box_class - 1] = torch.clamp(foreground_mask[batch_idx, box_class - 1],
                                                                            0, 1)

                elif mask_type == 'box':
                    # Box-type mask: Mark all voxels within the box
                    min_x = torch.min(rotated_corners[:, 0])
                    max_x = torch.max(rotated_corners[:, 0])
                    min_y = torch.min(rotated_corners[:, 1])
                    max_y = torch.max(rotated_corners[:, 1])

                    # Mark all voxels within the box
                    foreground_mask[batch_idx, box_class - 1, min_y:max_y + 1, min_x:max_x + 1] = 1

        return foreground_mask

    def visualize_foreground_mask_on_features(self, spatial_features, foreground_mask):
        """
        Visualize the foreground mask on the spatial features.

        Args:
            spatial_features: torch.Tensor of shape (B, C, H, W)
            foreground_mask: torch.Tensor of shape (B, C, H, W)
        """
        B, C, H, W = spatial_features.shape

        num_features_to_visualize = 2  # You can adjust this number as needed

        fig, axs = plt.subplots(num_features_to_visualize, 2, figsize=(20, 10))
        axs = axs.flatten()  # Convert axs to 1D array for easy indexing

        for i in range(min(num_features_to_visualize, C)):
            # Select the first batch's feature map and corresponding mask
            feature_map = spatial_features[1, i, :, :].cpu().detach().numpy()
            mask = foreground_mask[1, i, :, :].cpu().detach().numpy()

            # Plot the feature map
            axs[2 * i].imshow(feature_map, cmap='viridis')
            axs[2 * i].set_title(f'Feature Map {i}')
            axs[2 * i].axis('off')

            # Plot the mask
            axs[2 * i + 1].imshow(mask, cmap='viridis')
            axs[2 * i + 1].set_title(f'Foreground Mask {i}')
            axs[2 * i + 1].axis('off')

        plt.tight_layout()
        plt.show()

    def visualize_combined(self, spatial_features, foreground_mask, gt_boxes, frame_ids, reweighted_features):
        """
        Visualize the spatial features, foreground mask, and ground truth boxes in a single window.

        Args:
            reweighted_features: torch.Tensor of shape (B, C, H, W)
            spatial_features: torch.Tensor of shape (B, C, H, W)
            foreground_mask: torch.Tensor of shape (B, C, H, W)
            gt_boxes: torch.Tensor of shape (B, N, 8)
            frame_ids: list of frame IDs
        """
        B, C, H, W = spatial_features.shape
        N = gt_boxes.shape[1]

        num_features_to_visualize = 3  # You can adjust this number as needed

        fig, axs = plt.subplots(num_features_to_visualize, 4, figsize=(40, 10))
        axs = axs.flatten()  # Convert axs to 1D array for easy indexing
        def normalize_tensor(tensor):
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            return (tensor-min_val)/(max_val-min_val)

        for i in range(min(num_features_to_visualize, C)):
            # Select the first batch's feature map, mask, and ground truth boxes
            feature_map = normalize_tensor(spatial_features[0, i, :, :]).cpu().detach().numpy()
            reweighted_feature_map = normalize_tensor(reweighted_features[0, i, :, :]).cpu().detach().numpy()
            mask = foreground_mask[0, i, :, :].cpu().detach().numpy()
            boxes = gt_boxes[0].cpu().detach().numpy()
            fig.suptitle(f'Batch {0} - Frame ID: {frame_ids[0]}')

            # Plot the feature map
            axs[4 * i].imshow(feature_map, cmap='viridis')
            axs[4 * i].set_title(f'Feature Map {i}')
            axs[4 * i].axis('off')

            # Plot the reweighted feature map
            axs[4 * i + 1].imshow(reweighted_feature_map, cmap='viridis')
            # for box in boxes:
            #     gt_x, gt_y, gt_dx, gt_dy, heading = box[0], box[1], box[3], box[4], box[6]
            #     bbox_x = (gt_x - self.point_cloud_range[0]) / self.voxel_size[0]
            #     bbox_y = (gt_y - self.point_cloud_range[1]) / self.voxel_size[1]
            #     bbox_dx = gt_dx / self.voxel_size[0]
            #     bbox_dy = gt_dy / self.voxel_size[1]
            #
            #     # Calculate the four corners of the bounding box
            #     corners = np.array([
            #         [bbox_x - bbox_dx / 2, bbox_y - bbox_dy / 2],
            #         [bbox_x + bbox_dx / 2, bbox_y - bbox_dy / 2],
            #         [bbox_x + bbox_dx / 2, bbox_y + bbox_dy / 2],
            #         [bbox_x - bbox_dx / 2, bbox_y + bbox_dy / 2]
            #     ])
            #
            #     # Rotate the corners around the center
            #     rotation_matrix = np.array([
            #         [np.cos(-heading), -np.sin(-heading)],
            #         [np.sin(-heading), np.cos(-heading)]
            #     ])
            #     rotated_corners = np.dot(corners - [bbox_x, bbox_y], rotation_matrix) + [bbox_x, bbox_y]
            #
            #     # Convert to integer indices
            #     rotated_corners = np.clip(rotated_corners, 0, [W - 1, H - 1]).astype(int)
            #
            #     polygon = Polygon(rotated_corners, closed=True, edgecolor='r', facecolor='none', linewidth=1)
            #     axs[4 * i + 1].add_patch(polygon)

            axs[4 * i + 1].set_title(f'Reweighted Feature Map {i}')
            axs[4 * i + 1].axis('off')

            # Plot the mask
            axs[4 * i + 2].imshow(mask, cmap='gray')
            axs[4 * i + 2].set_title(f'Foreground Mask {i}')
            axs[4 * i + 2].axis('off')

            # Plot the feature map with ground truth boxes
            axs[4 * i + 3].imshow(feature_map, cmap='viridis')
            for box in boxes:
                gt_x, gt_y, gt_dx, gt_dy, heading = box[0], box[1], box[3], box[4], box[6]
                bbox_x = (gt_x - self.point_cloud_range[0]) / self.voxel_size[0]
                bbox_y = (gt_y - self.point_cloud_range[1]) / self.voxel_size[1]
                bbox_dx = gt_dx / self.voxel_size[0]
                bbox_dy = gt_dy / self.voxel_size[1]

                # Calculate the four corners of the bounding box
                corners = np.array([
                    [bbox_x - bbox_dx / 2, bbox_y - bbox_dy / 2],
                    [bbox_x + bbox_dx / 2, bbox_y - bbox_dy / 2],
                    [bbox_x + bbox_dx / 2, bbox_y + bbox_dy / 2],
                    [bbox_x - bbox_dx / 2, bbox_y + bbox_dy / 2]
                ])

                # Rotate the corners around the center
                rotation_matrix = np.array([
                    [np.cos(-heading), -np.sin(-heading)],
                    [np.sin(-heading), np.cos(-heading)]
                ])
                rotated_corners = np.dot(corners - [bbox_x, bbox_y], rotation_matrix) + [bbox_x, bbox_y]

                # Convert to integer indices
                rotated_corners = np.clip(rotated_corners, 0, [W - 1, H - 1]).astype(int)

                polygon = Polygon(rotated_corners, closed=True, edgecolor='r', facecolor='none', linewidth=1)
                axs[4 * i + 3].add_patch(polygon)

            axs[4 * i + 3].set_title(f'Feature Map with GT Boxes {i}')
            axs[4 * i + 3].axis('off')

        plt.tight_layout()
        plt.show()


# version 1 not concentrate and res attention

    def forward(self, data_dict, **kwargs):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']  # torch.Size([4, 64,496, 432])
        # visualize_tensor(spatial_features)
        # print("Max value in spatial_features:", torch.max(spatial_features).item()) # 1260.5780029296875

        frame_ids = data_dict['frame_id']

        gt_boxes = data_dict['gt_boxes']

        # self.visualize_gt_boxes(spatial_features, gt_boxes, self.voxel_size, self.point_cloud_range, frame_ids)
        match_box_to_bev = self.match_box_to_bev(spatial_features, gt_boxes).float()  # torch.Size([4, 3,496, 432])
        # print(torch.max(match_box_to_bev)) # 1.0
        # self.visualize_foreground_mask_on_features(spatial_features,match_box_to_bev)
        reweighted_features, probability_map = self.sce(spatial_features)  # [4, 64, 496, 432] [4, 3,496, 432]

        self.visualize_combined(spatial_features, match_box_to_bev, gt_boxes, frame_ids, reweighted_features)

        data_dict.update({'spatial_features': spatial_features}) # [4, 64,496, 432]
        data_dict.update({'match_box_to_bev': match_box_to_bev}) # gt [4, 3,496, 432]
        data_dict.update({'probability_map': probability_map}) # seg_mask [4, 3,496, 432]

        return data_dict


# version 2 not concentrate and not res attention(keep original spatial features)
# def forward(self, data_dict, **kwargs):
#     """
#     Args:
#         data_dict:
#             spatial_features
#     Returns:
#     """
#     spatial_features = data_dict['spatial_features']  # torch.Size([4, 64,496, 432])
#     # visualize_tensor(spatial_features)
#     # print("Max value in spatial_features:", torch.max(spatial_features).item()) # 1260.5780029296875
#
#     frame_ids = data_dict['frame_id']
#
#     gt_boxes = data_dict['gt_boxes']
#
#     # self.visualize_gt_boxes(spatial_features, gt_boxes, self.voxel_size, self.point_cloud_range, frame_ids)
#     match_box_to_bev = self.match_box_to_bev(spatial_features, gt_boxes).float()  # torch.Size([4, 64,496, 432])
#     # aggressive
#     match_box_to_bev = self.conv_foreground(match_box_to_bev)
#     # self.visualize_foreground_mask_on_features(spatial_features,match_box_to_bev)
#     # M =    #self.softmax(match_box_to_bev)  torch.Size([4, 64,496, 432])
#     # print(probability_map)
#     # print(probability_map.shape)
#     probability_map = self.softmax(spatial_features)
#     # print(torch.max(probability_map))  # 0.9819...
#     probability_map = self.conv_foreground(probability_map)
#
#     reweighted_features = spatial_features  # * (1 + M)  # torch.Size([4, 64, 496, 432])
#     # self.visualize_combined(spatial_features, match_box_to_bev, gt_boxes, frame_ids, reweighted_features)
#
#     data_dict.update({'spatial_features': reweighted_features})
#     data_dict.update({'match_box_to_bev': match_box_to_bev})
#     data_dict.update({'probability_map': probability_map})
#
#     return data_dict