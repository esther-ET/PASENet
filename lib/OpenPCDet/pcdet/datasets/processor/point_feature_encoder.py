import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        参数:
            data_dict:
                points: (N, 3 + C_in)
                ...
        返回:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: 是否使用xyz作为点特征
                ...
        """
        # 根据编码类型对点进行编码
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        # 将use_lead_xyz添加到data_dict中
        data_dict['use_lead_xyz'] = use_lead_xyz

        # 如果需要过滤sweeps并且源特征列表中包含'timestamp'
        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            # 获取最大sweeps数
            max_sweeps = self.point_encoding_config.max_sweeps
            # 获取'timestamp'在源特征列表中的索引
            idx = self.src_feature_list.index('timestamp')
            # 将'timestamp'列四舍五入到小数点后两位
            dt = np.round(data_dict['points'][:, idx], 2)
            # 获取最大时间差
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt)) - 1, max_sweeps - 1)]
            # 过滤掉时间差大于max_dt的点
            data_dict['points'] = data_dict['points'][dt <= max_dt]

        # 返回更新后的data_dict
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True
