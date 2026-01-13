import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        # 保存传入的anchor配置
        self.anchor_generator_cfg = anchor_generator_config
        # anchor的范围，用于确定生成anchor的区域
        self.anchor_range = anchor_range
        # 从配置中提取每个anchor的尺寸
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        # 从配置中提取每个anchor的旋转角度
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        # 从配置中提取每个anchor的高度
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        # 检查是否需要对齐中心，默认不对齐
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        # 断言：确保每个anchor的尺寸、旋转、和高度的数量一致
        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        # 记录anchor集合的数量
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        # 断言：grid的尺寸和anchor集合的数量一致
        assert len(grid_sizes) == self.num_of_anchor_sets
        # 保存所有生成的anchor
        all_anchors = []
        # 保存每个位置生成的anchor数量
        num_anchors_per_location = []

        # 迭代每个网格尺寸、anchor尺寸、旋转角度和高度
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):

            # 计算每个位置anchor的数量（旋转角度、尺寸和高度的组合） 2*1*1
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))

            # 根据是否对齐中心，计算网格步幅和偏移量
            if align_center:
                # 计算x、y方向上的步长（如果对齐中心）x y z x y z
                #                                0 1 2 3 4 5
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                # 中心对齐时的偏移量
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                # 如果不对齐中心，使用不同的步长和偏移量计算方法
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                # 无需偏移
                x_offset, y_offset = 0, 0

            # 生成x方向的偏移量（步长范围内等间隔）
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            # 生成y方向的偏移量
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            # 生成z方向的高度偏移量
            z_shifts = x_shifts.new_tensor(anchor_height)

            # 获取anchor尺寸和旋转角度的数量
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            # 将anchor的旋转角度和尺寸转换为tensor
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)

            # 生成x、y、z方向的网格坐标
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])  # [x_grid, y_grid, z_grid]
            # 将x、y、z方向的偏移量堆叠到一起形成anchor
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            # 扩展anchor的尺寸，使其与anchor的尺寸数量相同
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            # 将anchor的尺寸与偏移量组合
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            # 扩展anchor以适应旋转角度的数量
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            # 将旋转角度添加到anchors中
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat(
                [*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [x, y, z, num_size, num_rot, 7]

            # 重新排列维度，使z在第一个维度上
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            # 调整高度，使其从中心对齐
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            # 将生成的anchor添加到列表中
            all_anchors.append(anchors)

        # 返回所有生成的anchors以及每个位置的anchor数量
        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    # 使用easydict库来简化配置的定义
    from easydict import EasyDict

    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],  # anchor尺寸
            'anchor_rotations': [0, 1.57],  # anchor旋转角度
            'anchor_heights': [0, 0.5]  # anchor的高度
        })
    ]

    # 创建AnchorGenerator对象，指定anchor范围和配置
    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )

    # 使用pdb调试器进行断点调试
    import pdb

    pdb.set_trace()

    # 生成anchor，传入网格尺寸
    A.generate_anchors([[188, 188]])

