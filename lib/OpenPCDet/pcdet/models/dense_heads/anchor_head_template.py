import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        # model_cfg：模型的配置信息。
        # num_class：目标类别的数量。
        # class_names：目标类别的名称列表。
        # grid_size：特征图的网格大小。
        # point_cloud_range：点云数据的范围。
        # predict_boxes_when_training：是否在训练时预测边界框。
        # model_cfg!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 将输入参数赋值给类的属性
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        # 从模型配置中获取目标分配器（target assigner）的配置信息，并根据配置信息创建盒子编码器（box coder） TARGET_ASSIGNER_CONFIG
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )
        # 从模型配置中获取锚框生成器（anchor generator）的配置信息，并根据配置信息生成锚框（anchors） ANCHOR_GENERATOR_CONFIG
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )

        # 将生成的锚框转移到 GPU 并赋值给类的属性。
        self.anchors = [x.cuda() for x in anchors]
        # 根据目标分配器的配置信息获取目标分配器（target assigner）
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)
        # 初始化一个空的字典（用于存储前向传播的结果），并根据模型配置中的损失配置信息构建损失函数。
        self.forward_ret_dict = {}     # 在传入head的过程中会被赋值的！！！！！！！！！！！！！
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    # 根据配置信息（anchor_generator_cfg）和网格大小（grid_size）、点云范围（point_cloud_range）-->
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        # 创建锚框生成器（AnchorGenerator)
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        # 计算特征图上网格的大小（feature_map_size），该特征图将用于生成锚框。
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        # 使用锚框生成器生成锚框列表（anchors_list）和每个位置的锚框数量列表（num_anchors_per_location_list）
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)
        # 如果锚框的维度不是7（默认为7），则对每个生成的锚框进行维度扩展，以匹配给定的维度（anchor_ndim）
        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors
        # 返回生成的锚框列表和每个位置的锚框数量列表
        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        # 如果目标分配器的名称是 'ATSS'，则创建一个 ATSSTargetAssigner 的实例，并传入相应的参数。
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        # 如果目标分配器的名称是 'AxisAlignedTargetAssigner'，则创建一个 AxisAlignedTargetAssigner 的实例，并传入相应的参数
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        # 返回创建的目标分配器对象
        return target_assigner

    def build_losses(self, losses_cfg):
        # self.add_module 是 nn.Module 类的方法之一，用于向模型中添加子模块。
        # 在这里，它被用来将损失函数作为子模块添加到当前模型中，以便模型可以跟踪并正确地管理这些损失函数。
        # 当你调用 self.add_module(name, module) 时，它会将一个子模块添加到当前模块中，并且可以通过 self.name 访问它。这样做有助于组织模型结构，并使模型的结构更加清晰。
        # cls_loss_func: 创建了一个 Sigmoid Focal Classification Loss，用于处理分类任务的损失计算
        # self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        # reg_loss_func: 创建了一个回归损失函数，通常使用 Smooth L1 Loss 或 Weighted Smooth L1 Loss。
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        # dir_loss_func: 创建了一个方向预测损失函数，通常使用加权交叉熵损失 Weighted Cross Entropy Loss
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        # 分配目标。这个过程涉及将每个锚框与最匹配的真实边界框进行匹配，然后生成一些标签和边界框偏移量，用于训练模型

        targets_dict = self.target_assigner.assign_targets(self.anchors, gt_boxes)
        # 包含分配的目标信息的字典 targets_dict，包含类别标签、边界框偏移量等
        return targets_dict

    def get_cls_layer_loss(self):
        # 从模型的 forward_ret_dict 中获取分类预测值 cls_preds 和分类标签 box_cls_labels。
        cls_preds = self.forward_ret_dict['cls_preds']     # refine_cls_preds
        # print("cls_preds_is:", cls_preds.shape)  # ([4, H, W, 18]
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # multiple yong de
        # print("box_cls_labels_is:", box_cls_labels.shape)  # ([4, H, W]

        batch_size = int(cls_preds.shape[0])
        # 计算正负样本的加权，以便在计算损失时使用。这里通过 cared 确保只计算有标签的锚框。
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # 计算正样本的回归权重。
        reg_weights = positives.float()
        # 如果是单类别任务，则将所有正样本的类别标签设置为1，因为这是一个类别无关的任务
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        # 计算正样本的加权系数，并确保分母不为零
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 根据加权情况，确定最终的分类目标标签。
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)  # 表示在最后一个维度上增加一个维度
        # 将分类目标标签转换为one_hot
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        # print("cls_preds.shape", cls_preds.shape) #4, 248, 216, 18
        # 调整预测值和目标值的形状
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        # print("cls_preds.shape", cls_preds.shape)
        # print("one_hot_targets.shape", one_hot_targets.shape)
        # 使用交叉熵损失函数计算分类损失
        #  zai shang mian you : self.add_module(
        #             'cls_loss_func',
        #             loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        #         )
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size
        # 根据配置文件中的权重对损失进行加权
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        # 准备用于 TensorBoard 可视化的损失字典
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        # 返回计算得到的分类损失和 TensorBoard 字典
        return cls_loss, tb_dict

        # my refine_cls_loss

    def get_refine_cls_layer_loss(self):
        # 从模型的 forward_ret_dict 中获取分类预测值 cls_preds 和分类标签 box_cls_labels。
        refine_cls_preds = self.forward_ret_dict['Refine_cls_preds']     # refine_cls_preds
        box_cls_labels = self.forward_ret_dict['box_cls_labels']  # multiple yong de

        batch_size = int(refine_cls_preds.shape[0])
        # 计算正负样本的加权，以便在计算损失时使用。这里通过 cared 确保只计算有标签的锚框。
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # 计算正样本的回归权重。
        reg_weights = positives.float()
        # 如果是单类别任务，则将所有正样本的类别标签设置为1，因为这是一个类别无关的任务
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1
        # 计算正样本的加权系数，并确保分母不为零
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 根据加权情况，确定最终的分类目标标签。
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)  # 表示在最后一个维度上增加一个维度
        # 将分类目标标签转换为one_hot
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=refine_cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        # 调整预测值和目标值的形状
        refine_cls_preds = refine_cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        # 使用交叉熵损失函数计算分类损失
        #  zai shang mian you : self.add_module(
        #             'cls_loss_func',
        #             loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        #         )
        refine_cls_loss_src = self.cls_loss_func(refine_cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        refine_cls_loss = refine_cls_loss_src.sum() / batch_size
        # 根据配置文件中的权重对损失进行加权
        refine_cls_loss = refine_cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['refine_cls_weight']
        # 准备用于 TensorBoard 可视化的损失字典
        tb_dict = {
            'rpn_loss_cls': refine_cls_loss.item()
        }
        # 返回计算得到的分类损失和 TensorBoard 字典
        return refine_cls_loss, tb_dict

    def get_seg_layer_loss(self):
        """
        计算加权二分类交叉熵损失。

        Args:
            self.forward_ret_dict 中需要包含以下键值：
                - probability_map: 模型的预测概率图，形状为 [B, C, H, W]。
                - match_box_to_bev: 目标值（ground truth），形状为 [B, C, H, W]。
                - weights (可选): 权重值，形状为 [B, H, W]。如果未提供，则默认为全 1。

        Returns:
            loss: 加权二分类交叉熵损失。
        """
        # data_dict.update({'match_box_to_bev': match_box_to_bev})
        # data_dict.update({'probability_map': probability_map})

        probability_map = self.forward_ret_dict['probability_map']
        match_box_to_bev = self.forward_ret_dict['match_box_to_bev']
        # print("probability_map_is:", probability_map.shape)
        # print("match_box_to_bev_is:", match_box_to_bev.shape)
        # 如果没有提供权重，则默认为全 1
        if 'weights' in self.forward_ret_dict:
            weights = self.forward_ret_dict['weights']  # 形状: [B, H, W]
        else:
            weights = torch.ones_like(probability_map[:, 0, :, :])  # 形状: [B, H, W]

        # 计算加权二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            probability_map, match_box_to_bev, reduction='none'
        )  # 形状: [B, C, H, W]

        # 应用权重
        loss = loss * weights.unsqueeze(1)  # 广播权重到 [B, C, H, W]

        # 对所有维度取平均
        loss = loss.mean()
        tb_dict = {
            'seg_loss': loss.item()
        }
        return loss, tb_dict


    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        """
                   给盒子的方向维度添加正弦差异。
                   参数:
                       boxes1: 预测的盒子（N，...，7）或（N，7）
                       boxes2: 目标盒子（N，...，7）或（N，7）
                       dim: 方向维度的索引
                   返回:
                       boxes1: 添加了正弦差异的盒子
                       boxes2: 添加了正弦差异的盒子
                   """
        assert dim != -1        # 确保维度不为-1
        # 计算正弦差异
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        # 将正弦差异与原始盒子拼接
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        """
            计算方向回归的方向目标。

            参数:
                anchors: 锚框（N，...，7）
                reg_targets: 回归目标（N，...，7）
                one_hot: 是否将目标编码为one-hot向量
                dir_offset: 方向偏移
                num_bins: 方向分箱数

            返回:
                dir_cls_targets: 方向目标
            """
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        # 计算旋转后的地面实况角度
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        # 确保角度在[0，2*pi]范围内
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        # 将角度转换为箱索引
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()   # torch.floor 函数返回输入张量的向下取整值，即将张量中的每个元素都向下取整到最接近的整数。例如，torch.floor(2.9) 返回 2.0。
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)  #torch.clamp 函数将输入张量中的每个元素限制在指定范围内。它有三个参数：input 表示输入张量，min 表示限制的下界，max 表示限制的上界。如果输入张量的元素小于 min，则将其设置为 min；如果大于 max，则将其设置为 max；否则保持不变
        # if one_hot
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    # # 解码粗略框和先验锚框  path in lib/pointpillars_with_TANet/second/pytorch/models/loss_utils.py
    #     de_coarse_boxes = box_torch_ops.second_box_decode(coarse_box_preds, anchors)
    #
    #     # 解码GT和先验锚框
    #     de_gt_boxes = box_torch_ops.second_box_decode(reg_targets, anchors)
    #
    #     # 编码新的GT
    #     new_gt = box_torch_ops.second_box_encode(de_gt_boxes, de_coarse_boxes)
    #     box_preds, reg_targets = add_sin_difference(refine_box_preds, new_gt)  # 添加正弦误差
    #     # 计算细化定位损失
    #     refine_loc_losses = loc_loss_ftor(
    #         box_preds, reg_targets, weights=reg_weights)  # 定位损失

    def get_box_reg_layer_loss(self):
        # 获取输入：
        # box_preds: 预测的框参数，大小为(batch_size, num_anchors_per_location * num_anchors, box_dim)。
        # box_dir_cls_preds: 预测的框方向类别概率，大小为(batch_size, num_anchors_per_location * num_anchors,
        #                                               num_direction_bins)。
        # box_reg_targets: 真实框参数的标签，大小与box_preds相同。
        # box_cls_labels: 框的类别标签，用于区分正负样本，大小与box_preds相同。
        # batch_size: 当前批次中样本的数量。

        box_preds = self.forward_ret_dict['box_preds']  # cls_preds=refine_cls_preds, box_preds=refine_loc_preds, dir_cls_preds=refine_dir_preds
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)  # refine_dir_preds
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])
        # 计算正样本权重：
        # 通过box_cls_labels获取正样本标记。
        positives = box_cls_labels > 0
        # 计算每个正样本的权重，用于加权损失。
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 处理锚框,将锚框形状调整为 (batch_size, num_anchors, box_dim)
        if isinstance(self.anchors, list):
            if self.use_multihead:

                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        #    anchors.view(1, -1, anchors.shape[-1])：首先将锚框的形状调整为 (1, 锚框总数, 每个锚框的维度)，其中 anchors.shape[-1] 表示每个锚框的维度。
    # repeat(batch_size, 1, 1)：然后将这个形状的锚框复制 batch_size 次，沿着第 0 维度进行复制。这样就得到了形状为 (batch_size, 锚框总数, 每个锚框的维度) 的锚框张量，其中每个样本都有相同的一组锚框。
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
    # 首先，box_preds.view(batch_size, -1, box_preds.shape[-1]) 将预测框的形状调整为 (batch_size, 预测框总数, 每个预测框的维度)，其中 box_preds.shape[-1] 表示每个预测框的维度
    # 如果 not self.use_multihead，则执行 box_preds.shape[-1] // self.num_anchors_per_location 来确定每个预测框的维度应该是多少，即将预测框的维度分配给每个锚框的数量。
    # 最终得到形状为 (batch_size, 预测框总数, 每个预测框的维度) 的预测框张量，其中每个样本的预测框都根据锚框的数量进行了调整。
    #     print('box_preds_is:', box_preds.shape) # ([4, 124, 108, 168])
        #print('box_preds_is:', box_preds[-1]) #
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_preds.shape[-1])
        # print('bbbox_preds_is:', box_preds.shape)  # ([4, 321408, 7])  (321408)!!!!!!!!

        # sin(a - b) = sinacosb-cosasinb
        # 调用 add_sin_difference 函数，将框参数中的方向部分转换为正弦差异。这一步是为了处理方向的周期性
        # print('box_reg_targets_is:', box_reg_targets.shape)  # ([4, 321408, 7])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        # 计算位置损失：
        #
        #     使用平滑 L1 损失函数计算位置损失。
        #     将损失按批次求和，并除以批次大小得到平均损失。
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }
        # 计算方向损失：
        #
        #     如果存在 box_dir_cls_preds，表示模型预测了框的方向。
        #     根据锚框和真实框参数，计算方向类别标签。
        #     使用加权交叉熵损失函数计算方向损失。
        #     将损失按批次求和，并除以批次大小得到平均损失
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 返回损失：
            #
            #     将位置损失和方向损失（如果存在）相加作为总损失。
            #     返回总损失和记录的损失信息。
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict
    # 在这个框架里面，方向损失被合并在box损失里面了，有方向损失的时候直接加上就好

    def get_refine_box_reg_layer_loss(self):  # refine_loc_preds
        # 获取输入：
        # box_preds: 预测的框参数，大小为(batch_size, num_anchors_per_location * num_anchors, box_dim)。
        # box_dir_cls_preds: 预测的框方向类别概率，大小为(batch_size, num_anchors_per_location * num_anchors,
        #                                               num_direction_bins)。
        # box_reg_targets: 真实框参数的标签，大小与box_preds相同。
        # box_cls_labels: 框的类别标签，用于区分正负样本，大小与box_preds相同。
        # batch_size: 当前批次中样本的数量。

        refine_loc_preds = self.forward_ret_dict['Refine_loc_preds']     #  box_preds = self.forward_ret_dict['box_preds']
        #refine_dir_preds = self.forward_ret_dict.get('Refine_dir_preds', None)  # box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)

        box_reg_targets = self.forward_ret_dict['box_preds']       # box_reg_targets !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        batch_size = int(refine_loc_preds.shape[0])
        box_reg_targets = box_reg_targets.view(batch_size, -1, box_reg_targets.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_reg_targets.shape[-1])

        box_cls_labels = self.forward_ret_dict['box_cls_labels']
            # box_preds
        # 计算正样本权重：
        # 通过box_cls_labels获取正样本标记。
        positives = box_cls_labels > 0
        # 计算每个正样本的权重，用于加权损失。
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 处理锚框,将锚框形状调整为 (batch_size, num_anchors, box_dim)
        if isinstance(self.anchors, list):
            if self.use_multihead:

                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        #    anchors.view(1, -1, anchors.shape[-1])：首先将锚框的形状调整为 (1, 锚框总数, 每个锚框的维度)，其中 anchors.shape[-1] 表示每个锚框的维度。
    # repeat(batch_size, 1, 1)：然后将这个形状的锚框复制 batch_size 次，沿着第 0 维度进行复制。这样就得到了形状为 (batch_size, 锚框总数, 每个锚框的维度) 的锚框张量，其中每个样本都有相同的一组锚框。
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        #print('yuan_refine_loc_preds_is:', refine_loc_preds.shape) # ([4, 124, 108, 168])
        refine_loc_preds = refine_loc_preds.view(batch_size, -1,
                                                 refine_loc_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                                 refine_loc_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        # 调用 add_sin_difference 函数，将框参数中的方向部分转换为正弦差异。这一步是为了处理方向的周期性
        #print('refine_loc_preds_is:', refine_loc_preds.shape)  # ([4, 321408, 7])
        #55('box_reg_targets_is:', box_reg_targets.shape)  # ([4, 124, 108, 168])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(refine_loc_preds, box_reg_targets)
        # 计算位置损失：
        #
        #     使用平滑 L1 损失函数计算位置损失。
        #     将损失按批次求和，并除以批次大小得到平均损失。
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['refine_loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }
        # 计算方向损失：!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #
        #     如果存在 box_dir_cls_preds，表示模型预测了框的方向。
        #     根据锚框和真实框参数，计算方向类别标签。
        #     使用加权交叉熵损失函数计算方向损失。
        #     将损失按批次求和，并除以批次大小得到平均损失


        # if refine_dir_preds is not None:
        #     dir_targets = self.get_direction_target(
        #         anchors, box_reg_targets,
        #         dir_offset=self.model_cfg.DIR_OFFSET,
        #         num_bins=self.model_cfg.NUM_DIR_BINS
        #     )
        #     dir_logits = refine_dir_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
        #     weights = positives.type_as(dir_logits)
        #     weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 返回损失：
            #
            #     将位置损失和方向损失（如果存在）相加作为总损失。
            #     返回总损失和记录的损失信息。

            ####################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            # dir_loss = dir_loss.sum() / batch_size
            # dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['refine_dir_weight']
            # box_loss += dir_loss
            # tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict




    def get_loss(self, **kwargs):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        """
        生成预测的框。

                Args:
                    batch_size: 批量大小。
                    cls_preds: (N, H, W, C1) 分类预测结果。
                    box_preds: (N, H, W, C2) 边界框预测结果。
                    dir_cls_preds: (N, H, W, C3) 方向分类预测结果。

                Returns:
                    batch_cls_preds: (B, num_boxes, num_classes) 批量的类别预测结果。
                    batch_box_preds: (B, num_boxes, 7+C) 批量的边界框预测结果。
         """
        # # 如果锚框是列表形式，将它们连接成一个张量
        if isinstance(self.anchors, list):
            # 如果使用多头，需要对锚框进行重新排列
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        # 获取锚框的数量
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]#     anchors.shape[-1]：获取锚框张量的最后一个维度的大小，即锚框的特征数（通常是7，表示位置和尺寸信息）。
                                                                         #      anchors.view(-1, anchors.shape[-1])：将锚框张量重塑为两维张量，其中第一个维度为-1，表示未指定，系统会根据其他维度的大小自动计算。第二个维度为锚框的特征数。
        ## 将锚框复制到每个样本中                                  #       anchors.view(-1, anchors.shape[-1]).shape[0]：获取重塑后张量的第一个维度的大小，即锚框的数量。
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)  # -1的目的是将锚框的维度扩展为三维，以便与预测结果对齐
        # 调整预测的类别和边界框的形状
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        # 使用边界框解码器解码边界框
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        # 如果存在方向分类预测结果
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            # 调整方向分类预测的形状
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            # 从方向分类预测结果中获取标签
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
            # 计算角度偏移量和限制角度偏移量
            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            # 根据方向分类标签调整边界框的角度  ... 表示对张量的所有维度进行索引，除了被明确指定的维度(6)之外
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)
        # 如果使用的边界框编码器是PreviousResidualDecoder，则对角度进行额外处理
        # isinstance() 是 Python 内置函数，用于检查一个对象是否是指定类或类型的实例。
        # 在这个上下文中，isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder) 检查 self.box_coder 是否是 box_coder_utils.PreviousResidualDecoder 类的实例。
        # 如果是，则返回 True，否则返回 False。这样的检查通常用于在代码中根据对象的类型执行不同的操作。
        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds



    def forward(self, **kwargs):
        # seg_pred = kwargs.get('seg_pred')
        # match_voxels_to_boxes = kwargs.get('match_voxels_to_boxes')
        # # if seg_pred is not None and match_voxels_to_boxes is not None:
        # #     seg_loss = self.get_seg_layer_loss()

        raise NotImplementedError
