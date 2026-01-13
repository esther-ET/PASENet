import torch
from torch import nn
import numpy as np
from lib.pointpillars_with_TANet.torchplus.tools import change_default_args
from lib.pointpillars_with_TANet.torchplus.nn.modules.common import Empty, Sequential
from lib.pointpillars_with_TANet.torchplus.nn.modules.normalization import GroupNorm
from .anchor_head_template import AnchorHeadTemplate
from lib.deformable_conv import DeformConv2d
import torch.nn.functional as F
from lib.OpenPCDet.pcdet.models.backbones_2d.seg_mask import Segmask

# =================================maybe wrong !!!!!==================================================================================
# cfg = {
#                    # 使用已发布的模型：将 USE_PACA_WEIGHT 设置为 False。
#                                                      #     训练新模型：将 USE_PACA_WEIGHT 设置为 True。
#
#     'PSA': {
#         'C_Bottle': 128,     #   256
#         'C_Reudce': 32
#     },
# }
#
#
# class TANetHead1(AnchorHeadTemplate):                                          #class PSA(nn.Module):
#     def __init__(self,  # 初始化方法，用于定义类的属性和初始化操作
#                  model_cfg, input_channels, class_names, grid_size, point_cloud_range, voxel_size,
#                  use_norm=True,  # 是否使用标准化，默认为True
#                  num_class=3,  # 类别数量，默认为2  len(class_names)
#                  layer_nums=[3, 5, 5],  # 不同层的卷积核数量列表，默认为[3, 5, 5]
#                  layer_strides=[2, 2, 2] ,  # 不同层的步长列表，默认为[2, 2, 2]     you rpn [1 2 2]
#                  num_filters=[64, 128, 256],  # 不同层的卷积核数目列表，默认为[64, 128, 256]!!!!!!!!!!!!
#                  upsample_strides=[1, 2, 4],  # 上采样步长列表，默认为[1, 2, 4]
#                  num_upsample_filters=[256, 256, 256],  # 上采样后的卷积核数目列表，默认为[256, 256, 256]
#                  num_input_filters=64,  # 输入特征的通道数，默认为128      384
#                  num_dilation=[1, 2, 4, 1, 1, 1],  # num_dilation=[2, 2, 2]
#                  num_padding = [2, 1, 2, 2, 2, 1],
#                  num_padding_3 = [2, 1, 2, 2, 2, 2],                 # [1, 2, 2, 2, 2, 2]
#                  num_anchors_per_location=2,  # 每个位置 的锚点数量，默认为2    kuo da 12 bei!!!!  gai cheng guo 24
#                  encode_background_as_zeros=True,  # 是否将背景编码为零，默认为True
#                  use_direction_classifier=True,  # 是否使用方向分类器，默认为True
#                  use_groupnorm=False,  # 是否使用GroupNorm，默认为False
#                  num_groups=32,  # GroupNorm的分组数，默认为32
#                  use_bev=False,  # 是否使用BEV，默认为False
#                  box_code_size=7,  # 包围盒编码的大小，默认为7
#                  name='psa',   # 类的名称，默认为'psa'
#                  predict_boxes_when_training=True, **kwargs
#
#                  ):
#
#         """
#         :param use_norm:
#         :param num_class:
#         :param layer_nums:
#         :param layer_strides:
#         :param num_filters:
#         :param upsample_strides:
#         :param num_upsample_filters:
#         :param num_input_filters:
#         :param num_anchors_per_location:
#         :param encode_background_as_zeros:
#         :param use_direction_classifier:
#         :param use_groupnorm:
#         :param num_groups:
#         :param use_bev:
#         :param box_code_size:
#         :param name:
#         """
#         super().__init__(model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
#             predict_boxes_when_training=predict_boxes_when_training)            # super(PSA, self).__init__()调用了父类nn.Module的__init__方法，以确保正确初始化类的实例。
#         # self.num_anchors_per_location = num_anchors_per_location    ## 2  num_anchors_per_location ????????????????????????????????????????????
#         self._use_direction_classifier = use_direction_classifier  # True
#         self._use_bev = use_bev   # False
#         # self.segmask = Segmask(model_cfg, voxel_size, point_cloud_range, num_class)
#
#
#
#         # 这句代码是一个断言语句，用于在程序中检查一个条件是否为真。如果条件为假，则会引发 AssertionError 异常，从而中断程序的执行。
#         assert len(layer_nums) == 3
#         assert len(layer_strides) == len(layer_nums)
#         assert len(num_filters) == len(layer_nums)
#         assert len(upsample_strides) == len(layer_nums)
#         assert len(num_upsample_filters) == len(layer_nums)
#         factors = []
#         # ################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sum sum sum
#         if isinstance(self.num_anchors_per_location, int):
#             self.num_anchors_per_location = [self.num_anchors_per_location]
#         self.num_anchors_per_location = sum(self.num_anchors_per_location)
#
#         # # 遍历每个层的步长，计算每个层的步长的乘积是否能被上采样步长整除
#         # for i in range(len(layer_nums)):
#         #     assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
#         #     factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
#         # assert all([x == factors[0] for x in factors])
#         # # 如果使用标准化，则使用GroupNorm or 使用BatchNorm
#         if use_norm:   # True
#             if use_groupnorm:  # False
#                 BatchNorm2d = change_default_args(
#                     num_groups=num_groups, eps=1e-3)(GroupNorm)
#             else:
#                 BatchNorm2d = change_default_args(
#                     eps=1e-3, momentum=0.01)(nn.BatchNorm2d)    # default zhege
#             Conv2d = change_default_args(bias=False)(nn.Conv2d)
#             ConvTranspose2d = change_default_args(bias=False)(
#                 nn.ConvTranspose2d)
#         else:
#             BatchNorm2d = Empty
#             Conv2d = change_default_args(bias=True)(nn.Conv2d)
#             ConvTranspose2d = change_default_args(bias=True)(
#                 nn.ConvTranspose2d)
#
#         # note that when stride > 1, conv2d with same padding isn't
#         # equal to pad-conv2d. we should use pad-conv2d.
#         block2_input_filters = num_filters[0]
#         if use_bev:  # 如果使用BEV
#             # 构建BEV提取器，用于处理BEV特征
#             self.bev_extractor = Sequential(
#                 Conv2d(6, 32, 3, padding=1),  # 3x3的卷积核 6->32
#                 BatchNorm2d(32),  # 批标准化
#                 nn.ReLU(),  # 激活函数
#                 Conv2d(32, 64, 3, padding=1),  # 3x3的卷积核  32->64
#                 BatchNorm2d(64),  # 批标准化
#                 nn.ReLU(),  # 激活函数
#                 nn.MaxPool2d(2, 2),  # 最大池化层
#             )
#             block2_input_filters += 64  # 更新block2输入的通道数
#
#         # 定义block1，包括若干卷积层和批标准化层
#         self.block1 = Sequential(
#             nn.ZeroPad2d(1),  # 零填充
#             Conv2d(
#                 num_input_filters, num_filters[0], 3, stride=layer_strides[0], padding=0),  # 卷积层 , 参数num_input_filters表示输入通道数，num_filters[0]表示输出通道数，3表示卷积核大小为3x3，stride=layer_strides[0]表示卷积操作的步长，layer_strides[0]是从参数中传入的步长值。
#             BatchNorm2d(num_filters[0]),  # 批标准化 这是一个二维批标准化层，用于规范化卷积层的输出。num_filters[0]表示批标准化层的输入通道数，即前面卷积层的输出通道数。
#             nn.ReLU(),  # 激活函数
#         )
#
#         for i in range(layer_nums[0]):  # 遍历block1的所有卷积层,使用for循环添加了layer_nums[0]个相同结构的卷积层、批标准化层和ReLU激活函数层，以构建深度网络结构
#             self.block1.add(
#                 Conv2d(num_filters[0], num_filters[0], 3, padding=1))  # 卷积层
#             self.block1.add(BatchNorm2d(num_filters[0]))  # 批标准化
#             self.block1.add(nn.ReLU())  # 激活函数
#         #
#         # self.block1.add(DeformConv2d(num_filters[0], num_filters[0], 3, 1, padding=1, bias=False,
#         #                 offset_groups=1, with_mask=True))
#         # self.block1.add(BatchNorm2d(num_filters[0]))  # 批标准化
#         # self.block1.add(nn.ReLU())  # 激活函数
#
#         # for i in range(layer_nums[0]-1):  # 遍历block1的所有卷积层,使用for循环添加了layer_nums[0]个相同结构的卷积层、批标准化层和ReLU激活函数层，以构建深度网络结构
#         #     self.block1.add(
#         #         Conv2d(num_filters[0], num_filters[0], 3, padding=1))  # 卷积层
#         #
#         #     self.block1.add(BatchNorm2d(num_filters[0]))  # 批标准化
#         #     self.block1.add(nn.ReLU())  # 激活函数
#         # self.block1.add(
#         #     Conv2d(num_filters[0], num_filters[0], 3, padding=2, dilation=2))  # 卷积层
#         # self.block1.add(BatchNorm2d(num_filters[0]))  # 批标准化
#         # self.block1.add(nn.ReLU())  # 激活函数
#
#
#         # 定义block1的反卷积层
#         # 定义了一个反卷积层self.deconv1，它将通过转置卷积操作对block1的输出进行上采样。与普通的卷积层相比，转置卷积层可以实现图像的上采样，将特征图的大小扩大。
#         # 反卷积层的参数包括输出通道数num_upsample_filters[0]、卷积核大小和步长。BatchNorm2d和ReLU激活函数层用于规范化和激活反卷积层的输出
#         self.deconv1 = Sequential(
#             ConvTranspose2d(
#                 num_filters[0],
#                 num_upsample_filters[0],
#                 upsample_strides[0],
#                 stride=upsample_strides[0]),  # 反卷积层
#             BatchNorm2d(num_upsample_filters[0]),  # 批标准化
#             nn.ReLU(),   # 激活函数
#         )
#
#         # 定义block2，包括若干卷积层和批标准化层
#         self.block2 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(
#                 block2_input_filters, num_filters[1], 3, stride=layer_strides[1], padding=0),   # 卷积层
#
#             BatchNorm2d(num_filters[1]),      # 批标准化
#             nn.ReLU(),  # 激活函数
#         )
#         # for i in range(layer_nums[1]-1):  # 遍历block2的所有卷积层
#         #     self.block2.add(
#         #         Conv2d(num_filters[1], num_filters[1], 3, padding=1))  # 卷积层
#         #     self.block2.add(BatchNorm2d(num_filters[1]))  # 批标准化
#         #     self.block2.add(nn.ReLU())   # 激活函数
#         # self.block2.add(
#         #     Conv2d(num_filters[1], num_filters[1], 3, padding=2, dilation=2))  # 卷积层
#         # self.block2.add(BatchNorm2d(num_filters[1]))  # 批标准化
#         # self.block2.add(nn.ReLU())  # 激活函数
#
#         for i in range(layer_nums[1]):  # 遍历block2的所有卷积层
#             self.block2.add(
#                 Conv2d(num_filters[1], num_filters[1], 3, padding=1))  # 卷积层
#             self.block2.add(BatchNorm2d(num_filters[1]))  # 批标准化
#             self.block2.add(nn.ReLU())   # 激活函数
#         # self.block2.add(
#         #     DeformConv2d(num_filters[1], num_filters[1], 3, stride=1, padding=1, bias=False,offset_groups=1, with_mask=True)),
#         # self.block2.add(BatchNorm2d(num_filters[1]))  # 批标准化
#         # self.block2.add(nn.ReLU())  # 激活函数
#
#
#         self.deconv2 = Sequential(
#             ConvTranspose2d(
#                 num_filters[1],
#                 num_upsample_filters[1],
#                 upsample_strides[1],
#                 stride=upsample_strides[1]),   # 反卷积层
#             BatchNorm2d(num_upsample_filters[1]),  # 批标准化
#             nn.ReLU(),  # 激活函数
#         )
#
#         # 定义block3，包括若干卷积层和批标准化层
#         self.block3 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2], padding=0),  # 卷积层
#             BatchNorm2d(num_filters[2]),
#             nn.ReLU(),
#         )
#         for i in range(layer_nums[2]):
#             self.block3.add(
#                 Conv2d(num_filters[2], num_filters[2], 3, padding=1))
#             self.block3.add(BatchNorm2d(num_filters[2]))
#             self.block3.add(nn.ReLU())
#         # self.block3.add(DeformConv2d(num_filters[2], num_filters[2], 3, stride=1, padding=1, bias=False,
#         #                              offset_groups=1, with_mask=True))
#         # self.block3.add(BatchNorm2d(num_filters[2]))  # 批标准化
#         # self.block3.add(nn.ReLU())  # 激活函数
#
#         # for i in range(layer_nums[2]-1):
#         #     self.block3.add(
#         #         Conv2d(num_filters[2], num_filters[2], 3, padding=1))
#         #     self.block3.add(BatchNorm2d(num_filters[2]))
#         #     self.block3.add(nn.ReLU())
#         # self.block3.add(
#         #     Conv2d(num_filters[2], num_filters[2], 3, dilation=2, padding=2))  # 卷积层  , dilation=2 padding=2
#         # self.block3.add(BatchNorm2d(num_filters[2]))  # 批标准化
#         # self.block3.add(nn.ReLU())  # 激活函数
#
#
#         self.deconv3 = Sequential(
#             ConvTranspose2d(
#                 num_filters[2],
#                 num_upsample_filters[2],
#                 upsample_strides[2],
#                 stride=upsample_strides[2]),
#             BatchNorm2d(num_upsample_filters[2]),
#             nn.ReLU(),
#         )
#         # ##################  anchor head
#         # 背景编码为零，默认为True
#         if encode_background_as_zeros:
#             num_cls = self.num_anchors_per_location * num_class  # 2*3
#         else:
#             num_cls = self.num_anchors_per_location * (num_class + 1)
#         # num_upsample_filters is the RPN
#         # num_upsample_filters是RPN的输出通道数，即block1、block2和block3的输出通道数之和
#         # 通过卷积层self.conv_cls处理num_upsample_filters
#         self.conv_cls = nn.Conv2d(num_upsample_filters[0]*3, self.num_anchors_per_location * self.num_class, 1)   # 72 for outchannels:num_cls=6 26784   3--> 13392
#         # 定义回归的卷积层  xia mian shi yin wei yao sum jia de !!!!!!!!!
#
#
#        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!       !!!!!!!!!!!!!!!!!
#         self.conv_box = nn.Conv2d(
#             num_upsample_filters[0]*3, self.num_anchors_per_location * self.box_coder.code_size, 1)  #self.num_anchors_per_location * box_code_size = 2*7=14   14*6=84
#         # anchor_head_single.py has
#         if use_direction_classifier:
#             self.conv_dir_cls = nn.Conv2d(
#                 num_upsample_filters[0]*3, self.num_anchors_per_location * 2, 1)
#
#
#
#     def get_loss(self, tb_dict=None):
#         tb_dict = {} if tb_dict is None else tb_dict
#         coarse_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
#
#         coarse_loss_box, tb_dict_3 = self.get_box_reg_layer_loss()
#
#         # Calculate segmentation loss
#         # seg_loss, tb_dict_5 = self.get_seg_layer_loss()
#         total_loss = coarse_loss_cls + coarse_loss_box #+ 2*seg_loss # 1   2   2
#         tb_dict.update(tb_dict_1)
#
#         tb_dict.update(tb_dict_3)
#
#         # tb_dict.update(tb_dict_5)
#
#         return total_loss, tb_dict
#
#     def init_weights(self):
#         # 初始化分类器卷积层的偏置，用于处理类别不平衡问题
#         pi = 0.01
#         nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#         # 初始化回归器卷积层的权重
#         nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
#
#     # Pyramid Sampling中既有downsample (max pooling)又有upsample (deconv)
#     # 目的是将RPN中3个block的输出调整到统一大小，从而又可以形成3个新的block，可以分别进行concat，
#
#
#
#
#     def forward(self, data_dict):
#
#
#         # 从输入数据字典中获取名为 'spatial_features_2d' 的空间特征
#         # spatial_features_2d = data_dict['spatial_features']
#         # print('spatial_features_2d_is', spatial_features_2d.shape)  # ([4, 384, 248, 216])
#         spatial_features_2d = data_dict['spatial_features']
#         # print('spatial_features_2d_is', spatial_features_2d.shape)  # ([4, 384, 248, 216])
#         # x: [N, C, H, W] 粗回归，在上面的浅蓝、浅绿、浅黄块块
#         x1 = self.block1(spatial_features_2d)
#         # print('x1_is', x1.shape)  # ([4, 64, 124, 108]  ([4, 64, 248, 216])
#
#         up1 = self.deconv1(x1)
#         # print('up1_is', up1.shape)  # ([4, 256, 124, 108] ([4, 256, 248, 216])
#         x2 = self.block2(x1)
#         # print('x2_is', x2.shape)  # ([4, 128, 62, 54] ([4, 128, 124, 108])
#         up2 = self.deconv2(x2)
#         # print('up2_is', up2.shape)  # ([4, 256, 124, 108] ([4, 256, 248, 216])
#         x3 = self.block3(x2)
#         # print('x3_is', x3.shape)  # ([4, 256, 31, 27] ([4, 256, 62, 54])
#         up3 = self.deconv3(x3)
#         # print('up3_is', up3.shape)  # ([4, 256, 124, 108]) ([4, 256, 248, 216])
#         # 124he108 you wen ti !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         # coarse_feat = torch.add([up1, up2, up3], dim=1)
#         coarse_feat = torch.cat([up1, up2, up3], dim=1)
#         # print('coarse_feat_is',coarse_feat.shape)  # ([4, 768, 248, 216])
#         box_preds = self.conv_box(coarse_feat)   # zhe li cong 768 dao 14, hou lai 14 dao houmian le suo yi wen ti zai 124 he 108
#         # print('box_preds_is',box_preds.shape)  # box_preds_is torch.Size([4, 42, 248, 216]) 6*7
#         cls_preds = self.conv_cls(coarse_feat)
#         # print('cls_preds_is',cls_preds.shape)  # cls_preds_is torch.Size([4, 18, 248, 216]) 6*3
#
#         # [N, C, y(H), x(W)]
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
#
#         # forward_ret_dict is inheriting in father
#         self.forward_ret_dict['cls_preds'] = cls_preds
#         self.forward_ret_dict['box_preds'] = box_preds
#
#         # self.forward_ret_dict['probability_map'] = data_dict['probability_map']
#         # self.forward_ret_dict['match_box_to_bev'] = data_dict['match_box_to_bev']
#
#
#
#         if self._use_direction_classifier:
#             dir_cls_preds = self.conv_dir_cls(coarse_feat)
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
#
#
#
#         if self.training:
#             targets_dict = self.assign_targets(
#                 gt_boxes=data_dict['gt_boxes']
#             )
#             # 将GT更新到模型的 forward_ret_dict 字典中
#             self.forward_ret_dict.update(targets_dict)
#             # print(data_dict)
#             # with open('/home/ubuntu/SWW/data_dict.txt', 'w') as f:
#             #     for key, value in data_dict.items():
#             #         if isinstance(value, torch.Tensor):
#             #             info = f"{key}: {value.shape}\n"
#             #         else:
#             #             info = f"{key}: {type(value)}\n"
#             #         # print(info.strip())
#             #         f.write(info)
#             #     f.write("\nFull data_dict:\n")
#             #     f.write(str(data_dict))
#
#
#
#
#         # 如果不处于训练模式或者需要在训练时进行框预测
#         if not self.training or self.predict_boxes_when_training:
#             # 生成预测的类别和框坐标
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=data_dict['batch_size'],
#                 cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#             )
#             # 将生成的类别和框坐标存储在数据字典中
#             data_dict['batch_cls_preds'] = batch_cls_preds
#             # print("bbbbbbbbbbbbatch_cls_preds_shape", batch_cls_preds.shape) # ([4, 321408, 3])
#             data_dict['batch_box_preds'] = batch_box_preds
#             # print("bbbbbbbbbbbbatch_box_preds_shape", batch_box_preds.shape) # ([4, 321408, 7])
#             # 将类别预测归一化标志设置为 False
#             data_dict['cls_preds_normalized'] = False
#             # return data_dict：返回更新后的数据字典。
#             # print(data_dict)
#             # with open('/home/ubuntu/SWW/test_data_dict.txt', 'w') as f:
#             #     for key, value in data_dict.items():
#             #         if isinstance(value, torch.Tensor):
#             #             info = f"{key}: {value.shape}\n"
#             #         else:
#             #             info = f"{key}: {type(value)}\n"
#             #         # print(info.strip())
#             #         f.write(info)
#             #     f.write("\nFull data_dict:\n")
#             #     f.write(str(data_dict))
#
#
#         return data_dict

# =================================================================================================================================

class TANetHead1(AnchorHeadTemplate):
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

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        coarse_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        coarse_loss_box, tb_dict_3 = self.get_box_reg_layer_loss()

        # seg_loss, tb_dict_5 = self.get_seg_layer_loss()
        # total_loss = coarse_loss_cls + coarse_loss_box + seg_loss  # 1   2   1
        total_loss = coarse_loss_cls + coarse_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_3)
        # tb_dict.update(tb_dict_5)

        return total_loss, tb_dict

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

        # self.forward_ret_dict['probability_map'] = data_dict['probability_map']
        # self.forward_ret_dict['match_box_to_bev'] = data_dict['match_box_to_bev']

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


# class TANetHead1(AnchorHeadTemplate):
#
#     def __init__(self,  # 初始化方法，用于定义类的属性和初始化操作
#                  model_cfg, input_channels, class_names, grid_size, point_cloud_range,
#                  use_norm=True,  # 是否使用标准化，默认为True
#                  num_class=3,  # 类别数量，默认为2  len(class_names)
#                  layer_nums=[3, 5, 5],  # 不同层的卷积核数量列表，默认为[3, 5, 5]
#                  layer_strides=[1, 2, 2] ,  # 不同层的步长列表，默认为[2, 2, 2]
#                  num_filters=[64, 128, 256],  # 不同层的卷积核数目列表，默认为[128, 128, 256]!!!!!!!!!!!!
#                  upsample_strides=[1, 2, 4],  # 上采样步长列表，默认为[1, 2, 4]
#                  num_upsample_filters=[256, 256, 256],  # 上采样后的卷积核数目列表，默认为[256, 256, 256]
#                  num_input_filters=384,  # 输入特征的通道数，默认为128  384  64
#
#                  num_anchors_per_location=2,  # 每个位置 的锚点数量，默认为2    kuo da 12 bei!!!!  gai cheng guo 24
#                  encode_background_as_zeros=True,  # 是否将背景编码为零，默认为True
#                  use_direction_classifier=True,  # 是否使用方向分类器，默认为True
#                  use_groupnorm=False,  # 是否使用GroupNorm，默认为False
#                  num_groups=32,  # GroupNorm的分组数，默认为32
#                  use_bev=False,  # 是否使用BEV，默认为False
#                  box_code_size=7,  # 包围盒编码的大小，默认为7
#
#                  predict_boxes_when_training=True, **kwargs
#                  ):
#
#         """
#         :param use_norm:
#         :param num_class:
#         :param layer_nums:
#         :param layer_strides:
#         :param num_filters:
#         :param upsample_strides:
#         :param num_upsample_filters:
#         :param num_input_filters:
#         :param num_anchors_per_location:
#         :param encode_background_as_zeros:
#         :param use_direction_classifier:
#         :param use_groupnorm:
#         :param num_groups:
#         :param use_bev:
#         :param box_code_size:
#         :param name:
#         """
#         super().__init__(model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
#             predict_boxes_when_training=predict_boxes_when_training)            # super(PSA, self).__init__()调用了父类nn.Module的__init__方法，以确保正确初始化类的实例。
#         # self.num_anchors_per_location = num_anchors_per_location    ## 2  num_anchors_per_location ????????????????????????????????????????????
#         self._use_direction_classifier = use_direction_classifier  # True
#         self._use_bev = use_bev   # False
#         # 这句代码是一个断言语句，用于在程序中检查一个条件是否为真。如果条件为假，则会引发 AssertionError 异常，从而中断程序的执行。
#         assert len(layer_nums) == 3
#         assert len(layer_strides) == len(layer_nums)
#         assert len(num_filters) == len(layer_nums)
#         assert len(upsample_strides) == len(layer_nums)
#         assert len(num_upsample_filters) == len(layer_nums)
#         factors = []
#         # ################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! sum sum sum
#         if isinstance(self.num_anchors_per_location, int):
#             self.num_anchors_per_location = [self.num_anchors_per_location]
#         self.num_anchors_per_location = sum(self.num_anchors_per_location)
#
#         # # 遍历每个层的步长，计算每个层的步长的乘积是否能被上采样步长整除
#         # for i in range(len(layer_nums)):
#         #     assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
#         #     factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
#         # assert all([x == factors[0] for x in factors])
#         # # 如果使用标准化，则使用GroupNorm or 使用BatchNorm
#         if use_norm:   # True
#             if use_groupnorm:  # False
#                 BatchNorm2d = change_default_args(
#                     num_groups=num_groups, eps=1e-3)(GroupNorm)
#             else:
#                 BatchNorm2d = change_default_args(
#                     eps=1e-3, momentum=0.01)(nn.BatchNorm2d)    # default zhege
#             Conv2d = change_default_args(bias=False)(nn.Conv2d)
#             ConvTranspose2d = change_default_args(bias=False)(
#                 nn.ConvTranspose2d)
#         else:
#             BatchNorm2d = Empty
#             Conv2d = change_default_args(bias=True)(nn.Conv2d)
#             ConvTranspose2d = change_default_args(bias=True)(
#                 nn.ConvTranspose2d)
#
#         # note that when stride > 1, conv2d with same padding isn't
#         # equal to pad-conv2d. we should use pad-conv2d.
#         block2_input_filters = num_filters[0]
#         if use_bev:  # 如果使用BEV
#             # 构建BEV提取器，用于处理BEV特征
#             self.bev_extractor = Sequential(
#                 Conv2d(6, 32, 3, padding=1),  # 3x3的卷积核 6->32
#                 BatchNorm2d(32),  # 批标准化
#                 nn.ReLU(),  # 激活函数
#                 Conv2d(32, 64, 3, padding=1),  # 3x3的卷积核  32->64
#                 BatchNorm2d(64),  # 批标准化
#                 nn.ReLU(),  # 激活函数
#                 nn.MaxPool2d(2, 2),  # 最大池化层
#             )
#             block2_input_filters += 64  # 更新block2输入的通道数
#
#         # 定义block1，包括若干卷积层和批标准化层
#         self.block1 = Sequential(
#             nn.ZeroPad2d(1),  # 零填充
#             Conv2d(
#                 num_input_filters, num_filters[0], 3, stride=layer_strides[0], padding=0),  # 卷积层 , 参数num_input_filters表示输入通道数，num_filters[0]表示输出通道数，3表示卷积核大小为3x3，stride=layer_strides[0]表示卷积操作的步长，layer_strides[0]是从参数中传入的步长值。
#
#             BatchNorm2d(num_filters[0]),  # 批标准化 这是一个二维批标准化层，用于规范化卷积层的输出。num_filters[0]表示批标准化层的输入通道数，即前面卷积层的输出通道数。
#             nn.ReLU(),  # 激活函数
#         )
#
#
#         for i in range(layer_nums[0]-1):  # 遍历block1的所有卷积层,使用for循环添加了layer_nums[0]个相同结构的卷积层、批标准化层和ReLU激活函数层，以构建深度网络结构
#             self.block1.add(
#                 Conv2d(num_filters[0], num_filters[0], 3, padding=1))  # 卷积层
#
#             self.block1.add(BatchNorm2d(num_filters[0]))  # 批标准化
#             self.block1.add(nn.ReLU())  # 激活函数
#         self.block1.add(
#             Conv2d(num_filters[0], num_filters[0], 3, padding=2, dilation=2))  # 卷积层
#         self.block1.add(BatchNorm2d(num_filters[0]))  # 批标准化
#         self.block1.add(nn.ReLU())  # 激活函数
#
#         # 定义block1的反卷积层
#         # 定义了一个反卷积层self.deconv1，它将通过转置卷积操作对block1的输出进行上采样。与普通的卷积层相比，转置卷积层可以实现图像的上采样，将特征图的大小扩大。
#         # 反卷积层的参数包括输出通道数num_upsample_filters[0]、卷积核大小和步长。BatchNorm2d和ReLU激活函数层用于规范化和激活反卷积层的输出
#         self.deconv1 = Sequential(
#             ConvTranspose2d(
#                 num_filters[0],
#                 num_upsample_filters[0],
#                 upsample_strides[0],
#                 stride=upsample_strides[0]),  # 反卷积层
#             BatchNorm2d(num_upsample_filters[0]),  # 批标准化
#             nn.ReLU(),   # 激活函数
#         )
#
#         # 定义block2，包括若干卷积层和批标准化层
#         self.block2 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(
#                 block2_input_filters, num_filters[1], 3, stride=layer_strides[1], padding=0),   # 卷积层
#             # DeformConv2d(num_filters[1], num_filters[1], 3, stride=layer_strides[0], padding=1,
#             #              offset_groups=1, bias=False, with_mask=True),
#             BatchNorm2d(num_filters[1]),      # 批标准化
#             nn.ReLU(),  # 激活函数
#         )
#         for i in range(layer_nums[1]-1):  # 遍历block2的所有卷积层
#             self.block2.add(
#                 Conv2d(num_filters[1], num_filters[1], 3, padding=1))  # 卷积层
#             self.block2.add(BatchNorm2d(num_filters[1]))  # 批标准化
#             self.block2.add(nn.ReLU())   # 激活函数
#
#         self.block2.add(
#             Conv2d(num_filters[1], num_filters[1], 3, padding=2, dilation=2))  # 卷积层
#         self.block2.add(BatchNorm2d(num_filters[1]))  # 批标准化
#         self.block2.add(nn.ReLU())  # 激活函数
#
#         self.deconv2 = Sequential(
#             ConvTranspose2d(
#                 num_filters[1],
#                 num_upsample_filters[1],
#                 upsample_strides[1],
#                 stride=upsample_strides[1]),   # 反卷积层
#             BatchNorm2d(num_upsample_filters[1]),  # 批标准化
#             nn.ReLU(),  # 激活函数
#         )
#
#         # 定义block3，包括若干卷积层和批标准化层
#         self.block3 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2], padding=0),  # 卷积层
#             # DeformConv2d(num_filters[2], num_filters[2], 3, stride=layer_strides[0], padding=1,
#             #              offset_groups=1, bias=False, with_mask=True),
#             BatchNorm2d(num_filters[2]),
#             nn.ReLU(),
#         )
#         for i in range(layer_nums[2]-1):
#             self.block3.add(
#                 Conv2d(num_filters[2], num_filters[2], 3, padding=1))
#             self.block3.add(BatchNorm2d(num_filters[2]))
#             self.block3.add(nn.ReLU())
#         self.block3.add(
#             Conv2d(num_filters[2], num_filters[2], 3, dilation=2, padding=2))  # 卷积层  , dilation=2 padding=2
#         self.block3.add(BatchNorm2d(num_filters[2]))  # 批标准化
#         self.block3.add(nn.ReLU())  # 激活函数
#
#
#         self.deconv3 = Sequential(
#             ConvTranspose2d(
#                 num_filters[2],
#                 num_upsample_filters[2],
#                 upsample_strides[2],
#                 stride=upsample_strides[2]),
#             BatchNorm2d(num_upsample_filters[2]),
#             nn.ReLU(),
#         )
#         # ##################  anchor head
#         # 背景编码为零，默认为True
#         if encode_background_as_zeros:
#             num_cls = self.num_anchors_per_location * num_class  # 2*3
#         else:
#             num_cls = self.num_anchors_per_location * (num_class + 1)
#         # num_upsample_filters is the RPN
#         # num_upsample_filters是RPN的输出通道数，即block1、block2和block3的输出通道数之和
#         # 通过卷积层self.conv_cls处理num_upsample_filters
#         self.conv_cls = nn.Conv2d(sum(num_upsample_filters), self.num_anchors_per_location * self.num_class, 1)   # 72 for outchannels:num_cls=6 26784   3--> 13392
#         # 定义回归的卷积层  xia mian shi yin wei yao sum jia de !!!!!!!!!
#
#
#        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!       !!!!!!!!!!!!!!!!!
#         self.conv_box = nn.Conv2d(
#             sum(num_upsample_filters), self.num_anchors_per_location * self.box_coder.code_size, 1)  #self.num_anchors_per_location * box_code_size = 2*7=14   14*6=84
#         # anchor_head_single.py has
#         if use_direction_classifier:
#             self.conv_dir_cls = nn.Conv2d(
#                 sum(num_upsample_filters), self.num_anchors_per_location * 2, 1)
#
#
#         # ##################  refine
#         # (Coarse Regression最后concat得到的结果)需要经过1层conv得到blottle_conv，留着在之后Fine Regression中upsample后的element wise add中用
#         self.bottle_conv = nn.Conv2d(sum(num_upsample_filters), sum(num_upsample_filters)//3, 1)
#         # Pyramid Sampling模块，实际上Pyramid Sampling就是max pooling和deconv模块的组合
#         self.block1_dec2x = nn.MaxPool2d(kernel_size=2)   ### C=64
#         self.block1_dec4x = nn.MaxPool2d(kernel_size=4)   ### C=64
#
#         self.block2_dec2x = nn.MaxPool2d(kernel_size=2)  ### C=128                        ### upsample_strides=[1, 2, 4]
#         self.block2_inc2x = ConvTranspose2d(num_filters[1],num_filters[0]//2,upsample_strides[1],stride=upsample_strides[1])  ### C=32
#
#         self.block3_inc2x = ConvTranspose2d(num_filters[2],num_filters[1]//2,upsample_strides[1],stride=upsample_strides[1])    #### C=64
#         self.block3_inc4x = ConvTranspose2d(num_filters[2],num_filters[0]//2,upsample_strides[2],stride=upsample_strides[2])   #### C=32
#         # 这些是用于特征融合的1x1卷积层。它们将用于将不同尺度的特征图进行融合，以获得更丰富的语义信息。这些层的输入通道数分别是不同块的输出通道数之和，输出通道数与相应的块的输出通道数相同
#         self.fusion_block1 = nn.Conv2d(num_filters[0]+num_filters[0]//2+num_filters[0]//2, num_filters[0], 1) # 128
#         self.fusion_block2 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[1]//2, num_filters[1], 1) # 256
#         self.fusion_block3 = nn.Conv2d(num_filters[0]+num_filters[1]+num_filters[2], num_filters[2], 1) # 448
#         # conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#         # UP block de zuo bian decov
#         self.refine_up1 = Sequential(
#             ConvTranspose2d(num_filters[0],num_upsample_filters[0], upsample_strides[0],stride=upsample_strides[0]),
#             BatchNorm2d(num_upsample_filters[0]),
#             nn.ReLU(),
#         )
#         self.refine_up2 = Sequential(
#             ConvTranspose2d(num_filters[1],num_upsample_filters[1],upsample_strides[1],stride=upsample_strides[1]),
#             BatchNorm2d(num_upsample_filters[1]),
#             nn.ReLU(),
#         )
#         self.refine_up3 = Sequential(
#             ConvTranspose2d(num_filters[2],num_upsample_filters[2],upsample_strides[2], stride=upsample_strides[2]),
#             BatchNorm2d(num_upsample_filters[2]),
#             nn.ReLU(),
#         )
#
#         #######
#         C_Bottle = cfg['PSA']['C_Bottle']   # 128
#         C = cfg['PSA']['C_Reudce']          # 32
#         #
#         self.RF1 = Sequential(  # 3*3
#             Conv2d(C_Bottle*2, C, kernel_size=1, stride=1),
#             BatchNorm2d(C),
#             nn.ReLU(inplace=True),
#             Conv2d(C, C_Bottle*2, kernel_size=3, stride=1, padding=1, dilation=1),
#             BatchNorm2d(C_Bottle*2),
#             nn.ReLU(inplace=True),
#         )
#
#         self.RF2 = Sequential(  # 5*5
#             Conv2d(C_Bottle, C, kernel_size=3, stride=1, padding=1),  # C_Bottle
#             BatchNorm2d(C),
#             nn.ReLU(inplace=True),
#             Conv2d(C, C_Bottle, kernel_size=3, stride=1, padding=1, dilation=1),
#             BatchNorm2d(C_Bottle),
#             nn.ReLU(inplace=True),
#         )
#
#         self.RF3 = Sequential(  # 7*7
#             Conv2d(C_Bottle//2, C, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(C),
#             nn.ReLU(inplace=True),
#             Conv2d(C, C, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(C),
#             nn.ReLU(inplace=True),
#             Conv2d(C, C_Bottle//2, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(C_Bottle//2),
#             nn.ReLU(inplace=True),
#         )
#         # concat bang bian de yellow block  ## change change change 1 to 2
#         self.concat_conv1 = nn.Conv2d(num_filters[2], num_filters[2], kernel_size=3, padding=1)  ## kernel_size=3
#         self.concat_conv2 = nn.Conv2d(num_filters[2], num_filters[2], kernel_size=3, padding=1)
#         self.concat_conv3 = nn.Conv2d(num_filters[2], num_filters[2], kernel_size=3, padding=1)
#         # jian ce tou de gray block
#         self.refine_cls = nn.Conv2d(sum(num_upsample_filters),self.num_anchors_per_location * self.num_class, 1) #  num_cls 2*3=6 6*12=72
#         self.refine_loc = nn.Conv2d(sum(num_upsample_filters),self.num_anchors_per_location * self.box_coder.code_size, 1) # 168
#         if use_direction_classifier:
#             self.refine_dir = nn.Conv2d(sum(num_upsample_filters), self.num_anchors_per_location * 2, 1)
#
#
#
#     def get_loss(self,tb_dict=None):
#         tb_dict = {} if tb_dict is None else tb_dict
#         coarse_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
#         refine_loss_cls, tb_dict_2 = self.get_refine_cls_layer_loss()
#         coarse_loss_box, tb_dict_3 = self.get_box_reg_layer_loss()
#         refine_loss_box, tb_dict_4 = self.get_refine_box_reg_layer_loss()
#
#         total_loss = coarse_loss_cls + coarse_loss_box + (refine_loss_cls + refine_loss_box)*self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['refine_weight']
#         tb_dict.update(tb_dict_1)
#         tb_dict.update(tb_dict_2)
#         tb_dict.update(tb_dict_3)
#         tb_dict.update(tb_dict_4)
#         return total_loss, tb_dict
#
#     def init_weights(self):
#         # 初始化分类器卷积层的偏置，用于处理类别不平衡问题
#         pi = 0.01
#         nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#         # 初始化回归器卷积层的权重
#         nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
#
#     # Pyramid Sampling中既有downsample (max pooling)又有upsample (deconv)
#     # 目的是将RPN中3个block的输出调整到统一大小，从而又可以形成3个新的block，可以分别进行concat，
#
#     def forward(self, data_dict):
#         # 从输入数据字典中获取名为 'spatial_features_2d' 的空间特征
#         #
#         spatial_features_2d = data_dict['spatial_features_2d'] #输入,base_bev_backbone一致 ([4, 384, 248, 216])
#         # print('spatial_feature_2d', spatial_features_2d.shape)
#         # x: [N, C, H, W] 粗回归，在上面的浅蓝、浅绿、浅黄块块
#         x1 = self.block1(spatial_features_2d)
#         # print('x1_is', x1.shape)  # ([4, 64, 124, 108]  ([4, 64, 248, 216])
#
#         up1 = self.deconv1(x1)
#         #print('up1_is', up1.shape)  # ([4, 256, 124, 108] ([4, 256, 248, 216])
#         x2 = self.block2(x1)
#         #print('x2_is', x2.shape)  # ([4, 128, 62, 54] ([4, 128, 124, 108])
#         up2 = self.deconv2(x2)
#         #print('up2_is', up2.shape)  # ([4, 256, 124, 108] ([4, 256, 248, 216])
#         x3 = self.block3(x2)
#         #print('x3_is', x3.shape)  # ([4, 256, 31, 27] ([4, 256, 62, 54])
#         up3 = self.deconv3(x3)
#         #print('up3_is', up3.shape)  # ([4, 256, 124, 108]) ([4, 256, 248, 216])
#         # 124he108 you wen ti !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         coarse_feat = torch.cat([up1, up2, up3], dim=1)
#         # print('coarse_feat_is',coarse_feat.shape)  # ([4, 768, 124, 108]) ([4, 768, 248, 216])
#         box_preds = self.conv_box(coarse_feat)   # zhe li cong 768 dao 14, hou lai 14 dao houmian le suo yi wen ti zai 124 he 108
#         #print('box_preds_is',box_preds.shape)  # ([4, 14, 124, 108]) ([4, 14, 248, 216])
#         cls_preds = self.conv_cls(coarse_feat)
#         #print('cls_preds_is',cls_preds.shape)  # ([4, 72, 124, 108] ([4, 6, 248, 216])
#
#         # [N, C, y(H), x(W)]
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
#
#         # forward_ret_dict is inheriting in father
#         self.forward_ret_dict['cls_preds'] = cls_preds
#         self.forward_ret_dict['box_preds'] = box_preds
#
#         if self._use_direction_classifier:
#             dir_cls_preds = self.conv_dir_cls(coarse_feat)
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
#
#
#         ###############Refine:
#         blottle_conv = self.bottle_conv(coarse_feat)  # FB
#         # jin zi ta
#         x1_dec2x = self.block1_dec2x(x1)
#         x1_dec4x = self.block1_dec4x(x1)
#
#         x2_dec2x = self.block2_dec2x(x2)
#         x2_inc2x = self.block2_inc2x(x2)
#
#         x3_inc2x = self.block3_inc2x(x3)
#         x3_inc4x = self.block3_inc4x(x3)
#
#         # B1 de 3ge
#         concat_block1 = torch.cat([x1, x2_inc2x, x3_inc4x], dim=1)
#         fusion_block1 = self.fusion_block1(concat_block1)  # 128
#         # B2 de 3ge
#         concat_block2 = torch.cat([x1_dec2x, x2, x3_inc2x], dim=1)
#         fusion_block2 = self.fusion_block2(concat_block2)  # 256
#         # B3 de 3ge
#         concat_block3 = torch.cat([x1_dec4x, x2_dec2x, x3], dim=1)
#         fusion_block3 = self.fusion_block3(concat_block3) # 448
#         # 然后经过conv、upsample (deconv)，
#         # 7*7
#         refine_up1 = self.RF3(fusion_block1)  # 64--64
#         refine_up1 = self.refine_up1(refine_up1)  # 64--256
#         # print("refine_up1",len(refine_up1))
#         # 5*5
#         refine_up2 = self.RF2(fusion_block2)  # 128--128
#         refine_up2 = self.refine_up2(refine_up2)  # 128--256
#
#         # 3*3
#         refine_up3 = self.RF1(fusion_block3)  # 256--256
#         refine_up3 = self.refine_up3(refine_up3) # 256--256
#
#         # 得到大小(w, h)一致的特征图，接着用blottle_conv与这三个特征图做element wise add，
#         branch1_sum_wise = refine_up1 + blottle_conv
#         branch2_sum_wise = refine_up2 + blottle_conv
#         branch3_sum_wise = refine_up3 + blottle_conv
#         # 之后将它们concat
#         concat_conv1 = self.concat_conv1(branch1_sum_wise)
#         concat_conv2 = self.concat_conv2(branch2_sum_wise)  # conv2
#         concat_conv3 = self.concat_conv3(branch3_sum_wise)  # conv3
#         # 输出总特征
#         PSA_output = torch.cat([concat_conv1, concat_conv2, concat_conv3], dim=1)
#         # print('PSA_output_is', PSA_output.shape)  # ([4, 768, 124, 108]) ([4, 768, 248, 216])
#         refine_cls_preds = self.refine_cls(PSA_output)
#         refine_loc_preds = self.refine_loc(PSA_output)
#         # print('rrrrrrrrrrefine_loc_preds_is', refine_loc_preds.shape)  # ([4, 84, 124, 108]) ([4, 168, 248, 216])
#         refine_loc_preds = refine_loc_preds.permute(0, 2, 3, 1).contiguous()
#         refine_cls_preds = refine_cls_preds.permute(0, 2, 3, 1).contiguous()
#
#         self.forward_ret_dict["Refine_loc_preds"] = refine_loc_preds  # refine box_preds
#         self.forward_ret_dict["Refine_cls_preds"] = refine_cls_preds  # refine cls_preds
#
#         refine_dir_preds = None  # without this hou mian de the refine_dir_preds hui warning
#
#         if self._use_direction_classifier:
#             refine_dir_preds = self.refine_dir(PSA_output)
#             refine_dir_preds = refine_dir_preds.permute(0, 2, 3, 1).contiguous()
#             self.forward_ret_dict["Refine_dir_preds"] = refine_dir_preds
#
#         if self.training:
#             targets_dict = self.assign_targets(
#                 gt_boxes=data_dict['gt_boxes']
#             )
#             # 将GT更新到模型的 forward_ret_dict 字典中
#             self.forward_ret_dict.update(targets_dict)
#         # 如果不处于训练模式或者需要在训练时进行框预测
#         if not self.training or self.predict_boxes_when_training:
#             # 生成预测的类别和框坐标
#             batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                 batch_size=data_dict['batch_size'],
#                 cls_preds=refine_cls_preds, box_preds=refine_loc_preds, dir_cls_preds=refine_dir_preds
#             )
#             # 将生成的类别和框坐标存储在数据字典中
#             data_dict['batch_cls_preds'] = batch_cls_preds
#             data_dict['batch_box_preds'] = batch_box_preds
#             # 将类别预测归一化标志设置为 False
#             data_dict['cls_preds_normalized'] = False
#             # return data_dict：返回更新后的数据字典。
#         return data_dict
