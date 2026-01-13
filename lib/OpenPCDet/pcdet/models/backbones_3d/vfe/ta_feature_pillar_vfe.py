import torch
from .vfe_template import VFETemplate
from torch import nn
import torch.nn.functional as F
# from pcdet.models.backbones_3d.vfe.pillar_vfe import PillarVFE      # PillarVFE  get_paddings_indicator
# import numpy as np
from pcdet.models.backbones_3d.vfe.pillar_vfe import PFNLayer

# cfg = {
#     'TA': {
#         'INPUT_C_DIM': 10,       # dim_ca 9*9 and change 9 to 10
#         'NUM_POINTS_IN_VOXEL': 100,   # dim_pa and 100*8 change 100 to 32
#         'REDUCTION_R': 8,    # reduction_r
#         'BOOST_C_DIM': 64,   # or 32
#         'USE_PACA_WEIGHT': True                      # 使用已发布的模型：将 USE_PACA_WEIGHT 设置为 False。
#                                                      # 训练新模型,将 USE_PACA_WEIGHT 设置为 True。
#     },
#
# }
#
#
# class PALayer(nn.Module):
#     def __init__(self, dim_pa, reduction_pa):
#         super(PALayer, self).__init__()
#         #     第一个线性层 nn.Linear(dim_pa, dim_pa // reduction_pa) 的输入维度是 dim_pa，输出维度是 dim_pa // reduction_pa。这里的 dim_pa 是输入特征的维度，reduction_pa 是缩减比例，用于减小特征维度。
#         #     接着是一个 ReLU 激活函数，用于引入非线性。
#         #     最后一个线性层 nn.Linear(dim_pa // reduction_pa, dim_pa) 的输入维度是 dim_pa // reduction_pa，输出维度是 dim_pa，将特征维度恢复到原始的维度。
#         #
#         # 这个模型通常用于特征的降维和重建，可以作为一个特征变换器或者解码器使用。
#
#         self.fc = nn.Sequential(
#             nn.Linear(dim_pa, dim_pa // reduction_pa),  # ori: 100*8  (input, output)  return F.linear(input, self.weight, self.bias)
#             nn.ReLU(inplace=True),
#             nn.Linear(dim_pa // reduction_pa, dim_pa)
#         )
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(dim_pa, dim_pa // reduction_pa, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(dim_pa // reduction_pa),
#         #     nn.ReLU(),
#         #
#         #     nn.Conv2d(dim_pa // reduction_pa, dim_pa // reduction_pa, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(dim_pa // reduction_pa),
#         #     nn.ReLU(),
#         #
#         #     nn.Conv2d(dim_pa // reduction_pa, dim_pa,  kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(dim_pa // reduction_pa),
#         #     nn.ReLU(),
#         #
#         # )
#
#     def forward(self, x):
#         b, w, _ = x.size() # x:[9608, 100, 10]
#
#         y = torch.max(x, dim=2, keepdim=True)[0].view(b, w)  # max pooling
#         # print("Shape of y:", y.shape)  # Shape of y: torch.Size([9608, 100])  the first number is variable
#
#         out1 = self.fc(y).view(b, w, 1)      # 这句代码首先对 y 进行全连接操作 self.fc(y)，然后使用 view(b, w, 1) 将结果重新调整形状为 (b, w, 1)。
#         # for fc(y): 其中，weight 是一个形状为 (dim_pa // reduction_pa, dim_pa) 的权重矩阵，bias 是一个形状为 (dim_pa // reduction_pa,) 的偏置向量。
#         # 所以，当输入 y 通过 fc(y) 传递到这个线性层时，就会进行上述的线性变换操作。
#         # 数据的权重维数与pa ca维数不匹配的问题!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         return out1
#
#
# # Channel-wise attention for each voxel
# class CALayer(nn.Module):
#     def __init__(self, dim_ca, reduction_ca):
#         super(CALayer, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(dim_ca, dim_ca // reduction_ca),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim_ca // reduction_ca, dim_ca)
#         )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(dim_ca, dim_ca // reduction_ca, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(dim_ca // reduction_ca),
#         #     nn.ReLU(),
#         #
#         #     nn.Conv2d(dim_ca // reduction_ca, dim_ca // reduction_ca, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(dim_ca // reduction_ca),
#         #     nn.ReLU(),
#         #
#         #     nn.Conv2d(dim_ca // reduction_ca, dim_ca, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(dim_ca // reduction_ca),
#         #     nn.ReLU(),
#         #
#         # )
#
#     def forward(self, x):
#         b, _, c = x.size()
#         y = torch.max(x, dim=1, keepdim=True)[0].view(b, c) # 9198*10
#         y = self.fc(y).view(b, 1, c)
#
#         return y
#
#
# # Point-wise attention for each voxel
# class PACALayer(nn.Module):
#     def __init__(self, dim_ca, dim_pa, reduction_r):
#         super(PACALayer, self).__init__()
#         self.pa = PALayer(dim_pa,  dim_pa // reduction_r) #B W 1
#         self.ca = CALayer(dim_ca,  dim_ca // reduction_r) #B 1 C
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         pa_weight = self.pa(x)
#         ca_weight = self.ca(x)
#         paca_weight = torch.mul(pa_weight, ca_weight) # B W 1 and B 1 C = B W C
#         paca_normal_weight = self.sig(paca_weight)
#         out = torch.mul(x, paca_normal_weight) # B W C and B W C = B W C
#         return out, paca_normal_weight
#
# # Voxel-wise attention for each voxel
#
#
# class VALayer(nn.Module):
#     def __init__(self, c_num, p_num):
#         super(VALayer, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(c_num + 3, 1),
#             nn.ReLU(inplace=True)
#         )
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(c_num + 3, 1, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(1),
#         #     nn.ReLU(),
#         # )
#
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(p_num, 1),    # #######################
#             nn.ReLU(inplace=True)
#         )
#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(p_num, 1, kernel_size=1, stride=1),
#         #     nn.BatchNorm2d(1),
#         #     nn.ReLU(),
#         # )
#
#         self.sigmod = nn.Sigmoid()
#
#     def forward(self, voxel_center, paca_feat):
#         """
#         :param voxel_center: size (K,1,3)
#         :param PACA_Feat: size (K,N,C)
#         :return: voxel_attention_weight: size (K,1,1)
#
#         Args:
#
#         """
#         # 将 voxel_center 张量在第二个维度（索引为1）上重复 paca_feat.shape[1] 次，而在第一个和第三个维度（索引为0和2）上保持不变。
#         voxel_center_repeat = voxel_center.repeat(1, paca_feat.shape[1], 1)  # k,1,3--->k,N,3
#         # print(voxel_center_repeat.shape)
#         voxel_feat_concat = torch.cat([paca_feat, voxel_center_repeat], dim=-1)  # K,N,C---> K,N,(C+3)
#
#         feat_2 = self.fc1(voxel_feat_concat)  # K,N,(C+3)--->K,N,1
#         feat_2 = feat_2.permute(0, 2, 1).contiguous()  # K,N,1--->K,1,N
#
#         voxel_feat_concat = self.fc2(feat_2)  # K,1,N--->K,1,1
#
#         voxel_attention_weight = self.sigmod(voxel_feat_concat)  # K,1,1
#
#         return voxel_attention_weight # Q
#
#
# class VoxelFeature_TA(nn.Module):                    # two block na ge
#     def __init__(self, dim_ca=cfg['TA']['INPUT_C_DIM'], dim_pa=cfg['TA']['NUM_POINTS_IN_VOXEL'],
#                  reduction_r=cfg['TA']['REDUCTION_R'], boost_c_dim=cfg['TA']['BOOST_C_DIM'],
#                  use_paca_weight=cfg['TA']['USE_PACA_WEIGHT']):
#         super(VoxelFeature_TA, self).__init__()
#         self.PACALayer1 = PACALayer(dim_ca=dim_ca, dim_pa=dim_pa, reduction_r=reduction_r)
#         self.PACALayer2 = PACALayer(dim_ca=boost_c_dim, dim_pa=dim_pa, reduction_r=reduction_r)
#         self.voxel_attention1 = VALayer(c_num=dim_ca, p_num=dim_pa)
#         self.voxel_attention2 = VALayer(c_num=boost_c_dim, p_num=dim_pa)
#         self.use_paca_weight = use_paca_weight
#         self.FC1 = nn.Sequential(
#             nn.Linear(2*dim_ca, boost_c_dim),
#             nn.ReLU(inplace=True),
#         )
#         self.FC2 = nn.Sequential(
#             nn.Linear(boost_c_dim, boost_c_dim),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, voxel_center, x):
#         paca1, paca_normal_weight1 = self.PACALayer1(x)      # 111111111111111111
#         voxel_attention1 = self.voxel_attention1(voxel_center, paca1)
#         # if self.use_paca_weight:
#         #     paca1_feat = voxel_attention1 * paca1 * paca_normal_weight1 # Q*F1*M
#         # else:
#         #     paca1_feat = voxel_attention1 * paca1
#         paca1_feat = paca1 # F1
#         out1 = torch.cat([paca1_feat, x], dim=2)
#         out1 = self.FC1(out1)
#
#         paca2, paca_normal_weight2 = self.PACALayer2(out1)
#         voxel_attention2 = self.voxel_attention2(voxel_center, paca2)
#         # if self.use_paca_weight:
#         #     paca2_feat = voxel_attention2 * paca2 * paca_normal_weight2
#         # else:
#         #     paca2_feat = voxel_attention2 * paca2
#         paca2_feat = paca2 # F2
#         out2 = out1 + paca2_feat
#
#         out = self.FC2(out2)
#
#         return out
#
# # PillarFeature_TANet is modified from pointpillars.PillarFeatureNet
# # by introducing Triple Attention                                           # zhe duan mei yong shang ,dan shang mian de yong dao le
#
#
# class TAPillarVFE(VFETemplate):                       # PillarFeatureNet 增加了VoxelFeature_TA模块，作用是对points进行attention修正。
#     # def __init__(self,
#     #              num_input_features=4,
#     #              use_norm=True,
#     #              num_filters=(64,),
#     #              with_distance=False,
#     #              voxel_size=(0.16, 0.16, 4),       # []  0.2, 0.2, 4
#     #              pc_range=(0, -39.68, -3, 69.12, 39.68, 1 )):  #     0, -40, -3, 70.4, 40, 1
#     def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
#         super().__init__(model_cfg=model_cfg)
#         self.use_norm = self.model_cfg.USE_NORM
#         self.with_distance = self.model_cfg.WITH_DISTANCE
#         self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
#         num_point_features += 6 if self.use_absolute_xyz else 3
#         """
#         Pillar Feature Net with Tripe attention.
#         The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
#         similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
#         :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
#         :param use_norm: <bool>. Whether to include BatchNorm.
#         :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
#         :param with_distance: <bool>. Whether to include Euclidean distance to points.
#         :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
#         :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
#         """
#
#         """
#         带有三重注意力机制的柱状特征网络。
#         该网络准备柱状特征并通过 PFNLayers 执行前向传播。该网络的作用类似于 SECOND 中的 second.pytorch.voxelnet.VoxelFeatureExtractor。
#         :param num_input_features: <int>。输入特征的数量，可以是 x、y、z 或 x、y、z、r。
#         :param use_norm: <bool>。是否包括批归一化。
#         :param num_filters: (<int>: N)。N 个 PFNLayers 中每个的特征数量。
#         :param with_distance: <bool>。是否包括点到点的欧几里得距离。
#         :param voxel_size: (<float>: 3)。体素的尺寸，只使用 x 和 y 尺寸。
#         :param pc_range: (<float>: 6)。点云范围，只使用 x 和 y 的最小值。
#         """
#
#
#         if self.with_distance:
#             num_point_features += 1
#         self.num_filters = self.model_cfg.NUM_FILTERS # 64
#         assert len(self.num_filters) > 0
#
#         num_input_features = cfg['TA']['BOOST_C_DIM']
#
#         # Create PillarFeatureNet layers #num_filters 列表的第一个元素是输入特征的数量，后续元素是每个 PFNLayer 的特征数量
#         num_filters = [num_input_features] + list(self.num_filters) #[64,64]
#         # 实例化 VoxelFeature_TA 和 PFNLayers
#         self.VoxelFeature_TA = VoxelFeature_TA()
#         pfn_layers = []
#         for i in range(len(num_filters) - 1):
#             in_filters = num_filters[i]
#             out_filters = num_filters[i + 1]
#             if i < len(num_filters) - 2:
#                 last_layer = False
#             else:
#                 last_layer = True
#             pfn_layers.append(PFNLayer(in_filters, out_filters, self.use_norm, last_layer=last_layer))
#             # the if can also pfn_layers.append(
#             #                 PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
#             #             )
#         self.pfn_layers = nn.ModuleList(pfn_layers)
#
#         # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
#         # 需要柱状（体素）尺寸和 x/y 偏移量以计算柱状偏移量   openpcdet is more!!!!!!!   zzzzzzzzzzzzzzzz!!!!!!!!!!!!!!!
#         # self.vx = voxel_size[0]
#         # self.vy = voxel_size[1]
#         # self.x_offset = self.vx / 2 + pc_range[0]
#         # self.y_offset = self.vy / 2 + pc_range[1]
#         self.voxel_x = voxel_size[0]
#         self.voxel_y = voxel_size[1]
#         self.voxel_z = voxel_size[2]
#         self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
#         self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
#         self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
#
#     def get_output_feature_dim(self):
#         return self.num_filters[-1]
#
#     def get_paddings_indicator(self, actual_num, max_num, axis=0): # voxel_num_points, voxel_count, axis=0
#         # 在指定的轴上扩展 actual_num 的维度
#         actual_num = torch.unsqueeze(actual_num, axis + 1)
#
#         # 创建一个形状与 actual_num 相同的列表，并在指定轴上设置为 -1
#         max_num_shape = [1] * len(actual_num.shape)
#         max_num_shape[axis + 1] = -1
#
#         # 创建一个从 0 到 max_num-1 的张量，并将其形状调整为 max_num_shape
#         max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
#
#         # 比较 actual_num 和 max_num，生成一个布尔张量，指示哪些位置的 actual_num 大于 max_num
#         paddings_indicator = actual_num.int() > max_num
#
#         # 返回布尔张量 paddings_indicator
#         return paddings_indicator
#
#     def forward(self, batch_dict, **kwargs):
#
#         # 从输入的 batch_dict 中获取体素特征、体素点数和体素坐标
#         voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
#         # Find distance of x, y, and z from cluster center
#         # 计算 x、y、z 到聚类中心
#         points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
#         # 计算每个点云点到点云中心的距离
#         f_cluster = voxel_features[:, :, :3] - points_mean
#
#         # Find distance of x, y, and z from pillar center
#         # 计算 x、y、z 到柱状中心的距离
#         f_center = torch.zeros_like(voxel_features[:, :, :3])  # f_center = torch.zeros_like(features[:, :, :2])
#         # f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
#         # f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)
#         f_center[:, :, 0] = voxel_features[:, :, 0] - (
#                     coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
#         f_center[:, :, 1] = voxel_features[:, :, 1] - (
#                     coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
#         f_center[:, :, 2] = voxel_features[:, :, 2] - (
#                     coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
#         # features[:, :, :3]：表示取 features 张量的所有元素，并且取每个元素的前三个值。这通常用于获取张量中的前三个特征，比如坐标的 x、y、z 值。
#         # features[:, :, 3]：表示取 features 张量的所有元素，并且取每个元素的第四个值。这通常用于获取张量中的某个特定特征，比如某种属性的数值。
#
#         # 在这个上下文中，features 可能是一个形状为 (batch_size, num_points, num_features) 的张量，其中 num_features 表示每个点的特征数量。
#         # #因此，features[:, :, :3] 可能代表了每个点的坐标信息，而 features[:, :, 3] 可能代表了某种属性的数值。
#
#         # Combine together feature decorations
#         # 组合特征
#         # features_ls = [features, f_cluster, f_center]
#         if self.use_absolute_xyz:
#             features = [voxel_features, f_cluster, f_center]  # 3,  3,  3
#         # 否则，将体素特征中的通道维度之后的部分（通常是法向量或其他特征）、点云相对坐标和点云中心坐标拼接在一起作为最终特征
#         else:
#             features = [voxel_features[..., 3:], f_cluster, f_center]
#
#         if self.with_distance:
#             points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
#             features.append(points_dist)
#         # (num_voxels, max_points_per_voxel, 3+c) (num_voxels, max_points_per_voxel, 3) (num_voxels, max_points_per_voxel, 3)
#         features = torch.cat(features, dim=-1)  # num_voxels, max_points_per_voxel ,9+c=10
#
#         # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
#         # empty pillars remain set to zeros.
#
#         voxel_count = features.shape[1]
#         mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
#         # print(mask.shape) # [11635, 100]
#         # print(mask)
#         mask = torch.unsqueeze(mask, -1).type_as(voxel_features) #[11635, 100, 1]
#         # print(mask.shape)
#         features *= mask # torch.Size([11635, 100, 10])
#         # print(features.shape)
#         # 通过 VoxelFeature_TA 处理特征
#         features = self.VoxelFeature_TA(points_mean, features) #torch.Size([11635, 100, 64])
#         # print(features.shape) # 1111111111111111111111111111
#         # Forward pass through PFNLayers
#         # 通过 PFNLayers 进行前向传播
#         for pfn in self.pfn_layers:
#             features = pfn(features)  # torch.Size([11635, 1, 64])
#         # print(features.shape)  # torch.Size([11635, 1, 64])
#
#         features = features.squeeze(1)           #you xiang le yixia  yao fu zhi/ !!!!!    # features is squeezed in pointpillar_scatter.py by me
#         batch_dict['pillar_features'] = features  # torch.Size([11635, 1 ,64]) hope to be torch.Size([11635, 64])
#         # 将处理后的特征存储在 batch_dict 中，并返回
#         # with open('/home/ubuntu/SWW/batch_dict.txt', 'w') as f:
#         #     for key, value in batch_dict.items():
#         #         if isinstance(value, torch.Tensor):
#         #             info = f"{key}: {value.shape}\n"
#         #         else:
#         #             info = f"{key}: {type(value)}\n"
#         #         # print(info.strip())
#         #         f.write(info)
#         #     f.write("\nFull data_dict:\n")
#         #     f.write(str(batch_dict))
#
#         return batch_dict



class L2AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(L2AttentionLayer, self).__init__()
        # 定义 Query 和 Key 的卷积层
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False) # 64-->16
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        # 共享 Query 和 Key 的权重
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias
        # 定义 Value 的卷积层
        self.v_conv = nn.Conv1d(channels, channels, 1)
        # 定义特征变换的卷积层和 BatchNorm 层
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()  # 激活函数

    # def forward(self, x):
        # x是点云特征，xyz是点云坐标
        # 输入 x 的形状: [num_voxels, num_points, channels]
        # 将坐标信息与特征相加
        # num_voxels, num_points, channels = x.size()
        # print(x.shape)  # torch.Size([11937, 64, 100]
        # 计算 Query 和 Key
        # x_q = self.q_conv(x)
        # print(x_q.shape)  # torch.Size([11937, 16, 100]
        # x_k = self.k_conv(x)
        # print(x_k.shape)  # torch.Size([11937, 16, 100]
        #
        # # 计算 L2 范数距离
        #
        # diff = x_q - x_k  # torch.Size([11937, 16, 100]
        # l2_distance = torch.norm(diff, p=2, dim=-2, keepdim=True)  # [num_voxels, num_points, num_points]
        # print(l2_distance.shape)  # torch.Size([11937, 16, 100]
        # # 计算注意力权重
        # attention = F.softmax(-l2_distance, dim=-2)  # [num_voxels, num_points, num_points]
        # print(attention.shape)  # torch.Size([11937, 16, 100]
        # # 计算 Value
        # x_v = self.v_conv(x)  # [num_voxels, channels, num_points] --> torch.Size([11937, 64, 100])
        #
        # # 对 Value 进行加权求和
        # x_r = torch.einsum('bij,bkj->bik', attention, x_v)
        #
        # # 通过特征变换和残差连接
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        # x = x + x_r  # 残差连接
        #
        # # 恢复输出形状
        # x = x.permute(0, 2, 1)
    def forward(self, x):
        # 输入 x 的形状: [num_voxels, 16, 100]

        # print(x.shape)  # torch.Size([11937, 64, 100])

        # 计算 Query 和 Key
        x_q = self.q_conv(x)  # [num_voxels, 16, 100]
        # print(x_q.shape)  # torch.Size([11937, 16, 100])
        x_k = self.k_conv(x)  # [num_voxels, 16, 100]
        # print(x_k.shape)  # torch.Size([11937, 16, 100])

        # 计算 L2 范数距离（基于点的数量维度）
        x_q_expanded = x_q.unsqueeze(3)  # [num_voxels, 16, 100, 1]
        x_k_expanded = x_k.unsqueeze(2)  # [num_voxels, 16, 1, 100]
        diff = x_q_expanded - x_k_expanded  # [num_voxels, 16, 100, 100]
        l2_distance = torch.norm(diff, p=2, dim=1)  # [num_voxels, 100, 100]
        # print(l2_distance.shape)  # torch.Size([11937, 100, 100])

        # 计算注意力权重
        attention = F.softmax(-l2_distance, dim=-1)  # [num_voxels, 100, 100]
        # print(attention.shape)  # torch.Size([11937, 100, 100])

        # 计算 Value
        x_v = self.v_conv(x)  # [num_voxels, 64, 100]
        # print(x_v.shape)  # torch.Size([11937, 64, 100])

        # 对 Value 进行加权求和
        x_r = torch.einsum('bij,bkj->bik', attention, x_v)  # [num_voxels, 64, 100]
        # print(x_r.shape)  # torch.Size([11937, 100, 64])

        # 通过特征变换和残差连接
        x_r = x_r.permute(0, 2, 1) # [num_voxels, 64, 100]
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))  # [num_voxels, 64, 100]
        # print(x_r.shape)  # torch.Size([11937, 64, 100])

        # 残差连接
        x = x + x_r  # [num_voxels, 64, 100]
        # print(x.shape)  # torch.Size([11937, 64, 100])

        # 调整输出形状

        return x

class VoxelFeature_TA(nn.Module):
    def __init__(self,  channels=64):
        super(VoxelFeature_TA, self).__init__()

        # 定义 1D 卷积层和 BatchNorm 层
        self.conv1 = nn.Conv1d(10, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(2*channels, channels, kernel_size=1, bias=False)
        self.pos_xyz = nn.Conv1d(3, channels, 1)  # 用于将坐标信息映射到特征空间
        self.bn1 = nn.BatchNorm1d(channels)
        # 定义四个自注意力层
        self.a1 = L2AttentionLayer(channels)
        self.a2 = L2AttentionLayer(channels)
        # self.a3 = L2AttentionLayer(channels)
        # self.a4 = L2AttentionLayer(channels)

    def forward(self, x):
        batch_size, _, N = x.size()
        # print("11111111111111111", x.shape) # torch.Size([11589, 100, 10])
        # xyz = xyz.permute(0, 2, 1)  # 调整维度顺序：[batch_size, 3, num_points]
        # xyz = self.pos_xyz(xyz)  # 将坐标信息映射到特征空间
        # 通过第一个卷积层和 BatchNorm，激活函数为 ReLU
        x = x.permute(0, 2, 1)  # torch.Size([11589, 10, 100])
        x = F.relu(self.bn1(self.conv1(x)))  # 输出维度：[batch_size, channels, num_points] # torch.Size([11589, 64, 100])
        # 通过四个自注意力层
        x1 = self.a1(x)  # torch.Size([11937, 64, 100])

        x2 = self.a2(x1)  # torch.Size([11937, 64, 100])
        # x3 = self.a3(x2)
        # x4 = self.a4(x3)
        # 将四个自注意力层的输出拼接
        # x = torch.cat((x1, x2, x3, x4), dim=1)#64*4=256
        x = self.conv2(torch.cat((x1, x2), dim=1))  # 64*2=128 ---conv---> 64

        return x


class TAPillarVFE(VFETemplate):                       # PillarFeatureNet 增加了VoxelFeature_TA模块，作用是对points进行attention修正。
    # def __init__(self,
    #              num_input_features=4,
    #              use_norm=True,
    #              num_filters=(64,),
    #              with_distance=False,
    #              voxel_size=(0.16, 0.16, 4),       # []  0.2, 0.2, 4
    #              pc_range=(0, -39.68, -3, 69.12, 39.68, 1 )):  #     0, -40, -3, 70.4, 40, 1
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        """
        Pillar Feature Net with Tripe attention.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        if self.with_distance:
            num_point_features += 1
        self.num_filters = self.model_cfg.NUM_FILTERS # 64
        assert len(self.num_filters) > 0

        num_input_features = 64  # cfg['TA']['BOOST_C_DIM']

        # Create PillarFeatureNet layers #num_filters 列表的第一个元素是输入特征的数量，后续元素是每个 PFNLayer 的特征数量
        num_filters = [num_input_features] + list(self.num_filters) #[64,64]
        # 实例化 VoxelFeature_TA 和 PFNLayers
        self.VoxelFeature_TA = VoxelFeature_TA()
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, self.use_norm, last_layer=last_layer))
            # the if can also pfn_layers.append(
            #                 PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            #             )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # 需要柱状（体素）尺寸和 x/y 偏移量以计算柱状偏移量   openpcdet is more!!!!!!!   zzzzzzzzzzzzzzzz!!!!!!!!!!!!!!!
        # self.vx = voxel_size[0]
        # self.vy = voxel_size[1]
        # self.x_offset = self.vx / 2 + pc_range[0]
        # self.y_offset = self.vy / 2 + pc_range[1]
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0): # voxel_num_points, voxel_count, axis=0
        # 在指定的轴上扩展 actual_num 的维度
        actual_num = torch.unsqueeze(actual_num, axis + 1)

        # 创建一个形状与 actual_num 相同的列表，并在指定轴上设置为 -1
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1

        # 创建一个从 0 到 max_num-1 的张量，并将其形状调整为 max_num_shape
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)

        # 比较 actual_num 和 max_num，生成一个布尔张量，指示哪些位置的 actual_num 大于 max_num
        paddings_indicator = actual_num.int() > max_num

        # 返回布尔张量 paddings_indicator
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):

        # 从输入的 batch_dict 中获取体素特征、体素点数和体素坐标
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # Find distance of x, y, and z from cluster center
        # 计算 x、y、z 到聚类中心
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        # 计算每个点云点到点云中心的距离
        f_cluster = voxel_features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # 计算 x、y、z 到柱状中心的距离
        f_center = torch.zeros_like(voxel_features[:, :, :3])  # f_center = torch.zeros_like(features[:, :, :2])
        # f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        # f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                    coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                    coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                    coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        # features[:, :, :3]：表示取 features 张量的所有元素，并且取每个元素的前三个值。这通常用于获取张量中的前三个特征，比如坐标的 x、y、z 值。
        # features[:, :, 3]：表示取 features 张量的所有元素，并且取每个元素的第四个值。这通常用于获取张量中的某个特定特征，比如某种属性的数值。

        # 在这个上下文中，features 可能是一个形状为 (batch_size, num_points, num_features) 的张量，其中 num_features 表示每个点的特征数量。
        # #因此，features[:, :, :3] 可能代表了每个点的坐标信息，而 features[:, :, 3] 可能代表了某种属性的数值。

        # Combine together feature decorations
        # 组合特征
        # features_ls = [features, f_cluster, f_center]
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]  # 3,  3,  3
        # 否则，将体素特征中的通道维度之后的部分（通常是法向量或其他特征）、点云相对坐标和点云中心坐标拼接在一起作为最终特征
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        # (num_voxels, max_points_per_voxel, 3+c) (num_voxels, max_points_per_voxel, 3) (num_voxels, max_points_per_voxel, 3)
        features = torch.cat(features, dim=-1)  # num_voxels, max_points_per_voxel ,9+c=10

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # print(mask.shape) # [11635, 100]
        # print(mask)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features) #[11635, 100, 1]
        # print(mask.shape)
        features *= mask # torch.Size([11635, 100, 10])
        # print(features.shape)
        # 通过 VoxelFeature_TA 处理特征
        # features = self.VoxelFeature_TA(points_mean, features) #torch.Size([11635, 100, 64])
        features = self.VoxelFeature_TA(features)  # torch.Size([11635, 64, 100])
        features = features.permute(0, 2, 1)  # torch.Size([11635, 100, 64])
        # print("ffffffffffffffffffffff",features.shape) # 1111111111111111111111111111
        # Forward pass through PFNLayers
        # 通过 PFNLayers 进行前向传播
        for pfn in self.pfn_layers:
            features = pfn(features)  # torch.Size([11635, 1, 64])
        # print(features.shape)  # torch.Size([11635, 1, 64])

        features = features.squeeze(1)           #you xiang le yixia  yao fu zhi/ !!!!!    # features is squeezed in pointpillar_scatter.py by me
        batch_dict['pillar_features'] = features  # torch.Size([11635, 1 ,64]) hope to be torch.Size([11635, 64])
        # 将处理后的特征存储在 batch_dict 中，并返回
        # with open('/home/ubuntu/SWW/batch_dict.txt', 'w') as f:
        #     for key, value in batch_dict.items():
        #         if isinstance(value, torch.Tensor):
        #             info = f"{key}: {value.shape}\n"
        #         else:
        #             info = f"{key}: {type(value)}\n"
        #         # print(info.strip())
        #         f.write(info)
        #     f.write("\nFull data_dict:\n")
        #     f.write(str(batch_dict))
        return batch_dict
