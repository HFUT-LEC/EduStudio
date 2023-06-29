from ..common import EduDataTPL
import pandas as pd


class ECDDataTPL(EduDataTPL):
    default_cfg = {}


    def read_QQQ_group(self, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}_QQQ-group.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_QQQ_group = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['qqq_id:token', 'group_id:token'])
        feat_name2type, df_QQQ_group = cls._convert_df_to_std_fmt(df_QQQ_group)
        gp = df_QQQ_group.groupby("group_id")['qqq_id']
        gps = gp.groups
        gps_list = []
        for k, v in gps.items():
            #     print(list(v))
            gps_list.append(list(v))
        return feat_name2type, gps_list
