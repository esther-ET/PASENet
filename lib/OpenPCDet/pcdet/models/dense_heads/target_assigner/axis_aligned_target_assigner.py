import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils

class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        # 初始化目标分配器对象，传入模型配置、类别名称、框编码器以及是否匹配高度的标志位
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        # 获取模型配置中的锚点生成器配置
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        # 获取模型配置中的目标分配器配置
        self.box_coder = box_coder  # 负责编码/解码边界框（bounding boxes）的对象
        self.match_height = match_height  # 是否在三维高度上匹配锚框
        self.class_names = np.array(class_names)  # 物体检测的类别名称列表
        # 从锚点生成器配置中提取类别名称列表
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        # 获取正样本比例，如果配置中为负数则为None
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None # none
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE  # 样本数量 512
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES  # 是否根据样本数量进行归一化
        self.matched_thresholds = {}  # 匹配阈值字典
        self.unmatched_thresholds = {}  # 不匹配阈值字典
        # 为每个类别设置匹配和不匹配阈值
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)  # 是否使用多头检测（multi-head detection）
        # 以下代码注释掉，用于单独处理多头检测
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: 锚框列表 [(N, 7), ...]
            gt_boxes_with_classes: 带有类别信息的真实框 (B, M, 8)
        Returns:
            返回每个锚框对应的类别和回归目标
        """

        bbox_targets = []  # 用于存储边界框回归目标
        cls_labels = []  # 用于存储分类标签
        reg_weights = []  # 用于存储回归权重

        batch_size = gt_boxes_with_classes.shape[0]  # 批次大小
        gt_classes = gt_boxes_with_classes[:, :, -1]  # 从真实框中提取类别信息
        gt_boxes = gt_boxes_with_classes[:, :, :-1]  # 提取实际的边界框坐标
        # 对于每一个batch中的样本，逐个处理
        for k in range(batch_size):
            cur_gt = gt_boxes[k]  # 当前batch的suoyou真实框
            cnt = cur_gt.__len__() - 1  # 获取当前真实框的数量,de dao index
            while cnt > 0 and cur_gt[cnt].sum() == 0:  # xiang qian bian li,jiang 7 ge zuo biao xiang jia,找到非空的真实框
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]  # 过滤掉空的真实框
            cur_gt_classes = gt_classes[k][:cnt + 1].int()  # 获取当前样本的真实框类别

            target_list = []  # 用于存储当前样本的目标分配结果
            # 遍历每个锚框类别及其对应的锚框
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                # 根据锚框类别和真实框类别进行匹配，生成掩码
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name) #cur_gt_classes获取当前样本的真实框类别
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    # 对多头检测，调整锚框维度并展开
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]  # 获取特征图大小
                    anchors = anchors.view(-1, anchors.shape[-1])  # 展开锚框
                    selected_classes = cur_gt_classes[mask]  # 根据掩码筛选出匹配的类别

                # 分配单个锚框的目标
                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)


            # 如果使用了多头检测，合并各个头的目标
            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                # 对于单一头的检测，合并各个目标
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }

                # 将target_dict['box_reg_targets']中的所有元素沿着第二个维度（dim=-2）进行拼接
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])  # 存储边界框回归目标
            cls_labels.append(target_dict['box_cls_labels'])  # 存储分类标签
            reg_weights.append(target_dict['reg_weights'])  # 存储回归权重

        bbox_targets = torch.stack(bbox_targets, dim=0)  # 将边界框回归目标堆叠成张量
        cls_labels = torch.stack(cls_labels, dim=0)  # 将分类标签堆叠成张量
        reg_weights = torch.stack(reg_weights, dim=0)  # 将回归权重堆叠成张量
        all_targets_dict = {
            'box_cls_labels': cls_labels,  # 分类标签
            'box_reg_targets': bbox_targets,  # 边界框回归目标
            'reg_weights': reg_weights  # 回归权重
        }
        return all_targets_dict  # 返回所有目标字典

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        # 为单个锚框分配目标

        num_anchors = anchors.shape[0]  # 锚框数量
        num_gt = gt_boxes.shape[0]  # 真实框数量

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1  # 初始化标签为-1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1  # 初始化真实框ID为-1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0: #have GT and have anchors
            # 计算每个锚框与真实框的IoU（交并比）


            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            # 对于每个锚框，我们都知道了它与哪个真实框的IoU最大(真实框的索引)以及这个最大IoU值是多少。
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)#在dim=1，即每个锚框对应的所有真实框的IoU上找出最大值的索引。
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            # 找出每个（ground truth boxes）与所有锚框（anchors）的最大交并比（IoU）以及对应的  锚框索引。
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            #  这行代码是在获取每个真实框与其对应的最大IoU的锚框的IoU值。
            #  torch.arange(num_gt, device=anchors.device)生成一个与真实框数量相同的序列，用于索引每个真实框，
            #  gt_to_anchor_argmax则用于索引每个真实框对应的最大IoU的锚框。
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0  # 处理空的真实框
            gt_to_anchor_max[empty_gt_mask] = -1

            # 选出与真实框具有最大重叠的锚框
            # anchor_by_gt_overlap == gt_to_anchor_max 这个操作会返回一个布尔型的张量，其中值为 True 的位置表示对应的锚框与某个真实框的 IoU 是最大的。
            # 然后，.nonzero()[:, 0] 会找出这个布尔型张量中值为 True 的元素的索引，即找出与每个真实框具有最大 IoU 的锚框的索引
            # anchors_with_max_overlap 是一个一维张量，包含了与每个真实框具有最大 IoU 的锚框的索引。
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            # 获取那些与真实框具有最大IoU的锚框对应的真实框的索引
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            # 这行代码是为那些与真实框具有最大IoU的锚框分配类别标签。
            # gt_classes[gt_inds_force] 是获取这些锚框对应的真实框的类别，然后将这些类别赋值给对应的锚框。
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]  # 为这些锚框分配真实框类别
            # 这行代码是为那些与真实框具有最大IoU的锚框分配真实框的索引。
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            # 将IoU大于匹配阈值的锚框标记为正样本
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]  # 分配正样本类别
            gt_ids[pos_inds] = gt_inds_over_thresh.int()  # 分配正样本的真实框ID
            # 将IoU小于不匹配阈值的锚框标记为负样本
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)  # 所有锚框为负样本

        fg_inds = (labels > 0).nonzero()[:, 0]  # 获取所有正样本的索引

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)  # 计算正样本数量
            if len(fg_inds) > num_fg:
                # 如果正样本数量超过限制，随机禁用一部分正样本
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1  # 将多余的正样本标记为-1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()  # 计算需要的负样本数量
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0  # 将选中的负样本标记为0
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0  # 没有真实框或锚框时，所有锚框标记为负样本
            else:
                labels[bg_inds] = 0  # 将负样本标记为0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]  # 将与真实框有最大重叠的锚框标记为正样本

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))  # 初始化边界框回归目标为0
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 对正样本计算回归目标
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)  # 编码正样本的边界框回归目标

        reg_weights = anchors.new_zeros((num_anchors,))  # 初始化回归权重为0

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()  # 计算有效样本数量
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples  # 根据样本数量进行归一化
        else:
            reg_weights[labels > 0] = 1.0  # 正样本的回归权重设为1

        ret_dict = {
            'box_cls_labels': labels,  # 返回分类标签
            'box_reg_targets': bbox_targets,  # 返回边界框回归目标
            'reg_weights': reg_weights,  # 返回回归权重
        }
        return ret_dict  # 返回结果字典

