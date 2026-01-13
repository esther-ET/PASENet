import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES # 64
        self.nx, self.ny, self.nz = grid_size # 432 496 1
        # print(self.nx) #1408 when no dataset pre for pp
        # print(self.ny) #1600
        # print(self.nz) #40
        assert self.nz == 1
        #在forward方法中，首先从batch_dict中取出pillar特征和坐标。
        # 然后，通过遍历每个批次，创建一个空的空间特征张量（spatial_feature），并根据坐标将pillar特征放置到正确的位置。
        # 具体来说，batch_mask = coords[:, 0] == batch_idx这行代码是创建一个布尔掩码，用于选择当前批次的坐标和特征。
        # 然后，this_coords = coords[batch_mask, :]和pillars = pillar_features[batch_mask, :]这两行代码是使用这个掩码来选择当前批次的坐标和特征。
        # 接下来，indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]这行代码是计算每个特征应该放置在空间特征张量中的哪个位置。
        # 然后，spatial_feature[:, indices] = pillars这行代码是将特征放置到正确的位置。
        # 最后，所有批次的空间特征被堆叠起来，并调整形状以匹配期望的输出形状。这个结果被添加到batch_dict中并返回。

    def forward(self, batch_dict, **kwargs):
        # 取出特征和坐标
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        # .max()函数是取这一列中的最大值，也就是最大的批次编号。然后.int().item()将这个最大值转换为Python的整数类型。最后，+ 1是因为批次编号是从0开始的
        batch_size = coords[:, 0].max().int().item() + 1
        #print(coords[:, 0])  # tensor([0., 0., 0.,  ..., 3., 3., 3.], device='cuda:0')

        #print("batch_size:", batch_size) # 4=3+1

        for batch_idx in range(batch_size):  #  0, 1, 2, 3
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny, # 214272=1*432*496
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            # coords是一个二维数组，coords[:, 0]表示取coords的所有行的第一列的元素。
            # batch_idx是一个指定的索引值。
            # coords[:, 0] == batch_idx这个操作会返回一个布尔数组，数组的长度与coords[:, 0]相同。
            # 如果coords的某行第一列的值等于batch_idx，那么对应位置的布尔值就是True，否则就是False。
            # 最后，这个布尔数组被赋值给batch_mask。这种操作通常用于后续的条件索引，即选取数组中满足某种条件的元素
            #print("batch_idx is", batch_idx) # 0 1 2 3
            #print(coords[:, 0]) # tensor([0., 0., 0.,  ..., 3., 3., 3.], device='cuda:0')
            batch_mask = coords[:, 0] == batch_idx # torch.Size([6969]) biande
            # print("batch_mask is", batch_mask)
            # print(batch_mask.shape)# tensor([ True, True, True, ..., False, False, False], device='cuda:0'
            this_coords = coords[batch_mask, :] # torch.Size([2152, 4])  biande
            # print("this_coords is", this_coords)
            # print(this_coords.shape)
            # print(this_coords[:, 1]) #[0000000]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3] # torch.Size([2152])    biande
            # print(indices)
            indices = indices.type(torch.long)  # torch.Size([2152])    biande

            pillar_features = pillar_features.squeeze(1)  #  torch.Size([6969, 64])  biande

            # debug
            # print("Shape of batch_mask:", batch_mask.shape) # torch.Size([11635])
            # print("Shape of pillar_features:", pillar_features.shape) # torch.Size([11635, 1 ,64]) hope to be torch.Size([11635, 64])
            # print("Values in batch_mask:", batch_mask) # tensor([ True, True, True, ..., False, False, False], device='cuda:0')
            # print("Values in pillar_features:", pillar_features) #tensor([[[0.2313, 0.2454, 0.1885, ..., 0.2195, 0.2441, 1.1154]], [[0.2362, 0.2453, 0.1895, ..., 0.2192, 0.2447, 0.8723]], [[0.2364, 0.2456, 0.1889, ..., 0.2195, 0.2448, 3.2351]], ..., [[0.2386, 0.2455, 0.1886, ..., 0.2192, 0.2445, 2.5621]], [[0.2342, 0.2460, 0.1911, ..., 0.2197, 0.2443, 0.4466]], [[0.2389, 0.2456, 0.1882, ..., 0.2193, 0.2448, 6.1928]]], device='cuda:0', grad_fn=<MaxBackward0>)

            pillars = pillar_features[batch_mask, :] # torch.Size([2152, 64]) biande

            pillars = pillars.t()  # torch.Size([64, 2152]) biande

            spatial_feature[:, indices] = pillars  # ([64, 2152]) ([64, 1624]) ([64, 1881]) ([64, 1312])
            #print("spatial_feature is", spatial_feature[:, indices].shape)
            batch_spatial_features.append(spatial_feature)#  torch.Size([64, 214272])  bu bian

        #
        batch_spatial_features = torch.stack(batch_spatial_features, 0) # ([4, 64, 214272])  四个张量进行拼接并且在dim0形成新的维度

        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx) #([4, 64, 496, 432])

        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
