import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        # 计算每个位置的锚框数量
        self.num_anchors_per_location = sum(self.num_anchors_per_location) # 2+2+2
        # 创建用于分类的卷积层
        # Shape:
        #         - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        #         - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        # 创建用于回归的卷积层
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        # 如果模型配置中指定了使用方向分类器，则创建相应的卷积层；否则置为Non
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        # 初始化网络权重
        self.init_weights()

    # 初始化网络权重的方法
    def init_weights(self):
        # 初始化分类器卷积层的偏置，用于处理类别不平衡问题
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # 初始化回归器卷积层的权重
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # 从输入数据字典中获取名为 'spatial_features_2d' 的空间特征
        spatial_features_2d = data_dict['spatial_features_2d']
        # print("spatial_features_2d_shape", spatial_features_2d.shape)  #
        # 通过卷积层 self.conv_cls 处理空间特征，得到类别预测
        cls_preds = self.conv_cls(spatial_features_2d)
        # print("single_head_class_shape", cls_preds.shape)                 # 3*6=18
        # print("num_anchors_per_location", self.num_anchors_per_location)  # 6
        #print("num_class", self.num_class)  #
        # 通过卷积层 self.conv_box 处理空间特征，得到框坐标预测。
        box_preds = self.conv_box(spatial_features_2d)
        # print("single_head_box_shape", box_preds.shape)                   # 7*6=42

        # 对类别预测和框坐标预测进行维度重排，将通道维移到最后一个维度
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # 将类别预测和框坐标预测存储在模型的 forward_ret_dict 字典中
        self.forward_ret_dict['cls_preds'] = cls_preds
        # print("cls_preds_shape", cls_preds.shape)  # 3*6=18
        self.forward_ret_dict['box_preds'] = box_preds
        # 检查是否存在方向类别预测的卷积层
        if self.conv_dir_cls is not None:                                # dir classifier is used
            # 如果存在方向类别预测的卷积层，则通过该卷积层处理空间特征，得到方向类别预测
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            # print("single_head_dir_shape", dir_cls_preds.shape)         # 2*6=12
            # 对方向类别预测进行维度重排permute，将通道维移到最后一个维度contiguous-->lian xv de
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            # 将方向类别预测存储在模型的 forward_ret_dict 字典中
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        # 如果处于训练模式，则根据真实边界框生成目标字典,需要对每个先验框分配GT来计算loss
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            # 将GT更新到模型的 forward_ret_dict 字典中
            self.forward_ret_dict.update(targets_dict)
        # 如果不处于训练模式或者需要在训练时进行框预测
        if not self.training or self.predict_boxes_when_training:
            # 生成预测的类别和框坐标
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            # 将生成的类别和框坐标存储在数据字典中
            data_dict['batch_cls_preds'] = batch_cls_preds
            #print("bbbbbbbbbbbbatch_cls_preds_shape", batch_cls_preds.shape)  # 18
            data_dict['batch_box_preds'] = batch_box_preds
            #print("bbbbbbbbbbbbatch_box_preds_shape", batch_box_preds.shape)
            # 将类别预测归一化标志设置为 False
            data_dict['cls_preds_normalized'] = False
        # return data_dict：返回更新后的数据字典。
        return data_dict
