# 导入父类 Detector3DTemplate
from .detector3d_template import Detector3DTemplate


# 定义 PVRCNN 类，继承自 Detector3DTemplate 类
class PVRCNN(Detector3DTemplate):
    # 初始化方法，接受模型配置、类别数量和数据集三个参数
    def __init__(self, model_cfg, num_class, dataset):
        # 调用父类的初始化方法，传入模型配置、类别数量和数据集
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # 构建网络模块列表
        self.module_list = self.build_networks()

    # 前向传播方法，接受输入的 batch_dict，进行前向传播
    def forward(self, batch_dict):
        # 遍历网络模块列表，对 batch_dict 进行处理
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # 如果模型处于训练状态
        if self.training:
            # 调用 get_training_loss 方法计算训练损失
            loss, tb_dict, disp_dict = self.get_training_loss()
            # 构建返回字典，包括损失值
            ret_dict = {
                'loss': loss
            }
            # 返回损失值、TensorBoard 字典和展示字典
            return ret_dict, tb_dict, disp_dict
        # 如果模型处于推断状态
        else:
            # 调用 post_processing 方法进行后处理，得到预测结果和召回结果
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # 返回预测结果和召回结果
            return pred_dicts, recall_dicts

    # 计算训练损失的方法
    def get_training_loss(self):
        # 初始化展示字典
        disp_dict = {}
        # 调用 dense_head 网络模块的 get_loss 方法计算 RPN 的损失
        loss_rpn, tb_dict = self.dense_head.get_loss()
        # 调用 point_head 网络模块的 get_loss 方法计算点云头的损失
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        # 调用 roi_head 网络模块的 get_loss 方法计算 RCNN 的损失
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        # 计算总损失，包括 RPN 损失、点云头损失和 RCNN 损失
        loss = loss_rpn + loss_point + loss_rcnn
        # 返回总损失、TensorBoard 字典和展示字典
        return loss, tb_dict, disp_dict

