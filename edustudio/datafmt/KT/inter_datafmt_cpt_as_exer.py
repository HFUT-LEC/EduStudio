from .kt_inter_datafmt import KTInterDataFmt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


class KTInterDataFmtCptAsExer(KTInterDataFmt):
    def __init__(self, cfg, train_dict, val_dict, test_dict, feat2type, **kwargs):
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type, **kwargs)

    @classmethod
    def _read_data_from_divided(cls, cfg):
        # config
        train_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-train.inter.csv'
        val_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-val.inter.csv'
        test_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-test.inter.csv'
        sep = cfg.datafmt_cfg['seperator']
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']

        # read data
        train_headers = pd.read_csv(train_file_path, nrows=0).columns.tolist()
        test_headers = pd.read_csv(test_file_path, nrows=0).columns.tolist()
        inter_train_df = pd.read_csv(
            train_file_path, sep=sep, encoding='utf-8', usecols=set(train_headers) - set(exclude_feats)
        )
        inter_test_df = pd.read_csv(
            test_file_path, sep=sep, encoding='utf-8', usecols=set(test_headers) - set(exclude_feats)
        )
        inter_val_df = None
        if os.path.exists(val_file_path):
            val_headers = pd.read_csv(val_file_path, nrows=0).columns.tolist()
            inter_val_df = pd.read_csv(
                val_file_path, sep=sep, encoding='utf-8', usecols=set(val_headers) - set(exclude_feats)
            )

        # to standard fmt
        d1, inter_train_df = cls._convert_df_to_std_fmt(inter_train_df)
        d2, inter_val_df = cls._convert_df_to_std_fmt(inter_val_df)
        d3, inter_test_df = cls._convert_df_to_std_fmt(inter_test_df)
        feat2type = d1
        feat2type.update(d2)
        feat2type.update(d3)

        # sort
        if 'timestamp' in feat2type:
            inter_train_df = inter_train_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
            if inter_val_df is not None:
                inter_val_df = inter_val_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
            inter_test_df = inter_test_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

        feat2type_, df_Q = cls.read_Q_matrix(cfg)
        feat2type.update(feat2type_)
        
        # 1.重命名df、df_Q中exer_id字段为exer_id_old，将df_Q中exer_seq转换为集合，变换成新的exer_id
        # 2.每个df与df_Q left merge。删除每个df的旧exer_id字段
        df_Q.rename(columns={"exer_id": "exer_id_old"}, inplace=True)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].apply(lambda x: "_".join(list(map(lambda i: str(i), set(x)))))
        lbe = LabelEncoder()
        df_Q['cpt_seq'] = lbe.fit_transform(df_Q['cpt_seq'])
        df_Q.rename(columns={"cpt_seq": "exer_id"}, inplace=True)

        inter_train_df.rename(columns={'exer_id': 'exer_id_old'}, inplace=True)
        inter_train_df = inter_train_df.merge(df_Q, on='exer_id_old', how='left').reset_index(drop=True)
        inter_train_df.drop(labels=['exer_id_old'], axis=1, inplace=True)
        if inter_val_df is not None:
            inter_val_df.rename(columns={'exer_id': 'exer_id_old'}, inplace=True)
            inter_val_df = inter_val_df.merge(df_Q, on='exer_id_old', how='left').reset_index(drop=True)
            inter_val_df.drop(labels=['exer_id_old'], axis=1, inplace=True)
        inter_test_df.rename(columns={'exer_id': 'exer_id_old'}, inplace=True)
        inter_test_df = inter_test_df.merge(df_Q, on='exer_id_old', how='left').reset_index(drop=True)
        inter_test_df.drop(labels=['exer_id_old'], axis=1, inplace=True)

        return feat2type, inter_train_df, inter_val_df, inter_test_df

    @classmethod
    def _read_data_from_undivided(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}.inter.csv'
        headers = pd.read_csv(file_path, nrows=0).columns.tolist()
        exclude_feats = cfg.datafmt_cfg['inter_exclude_feat_names']
        inter_df = pd.read_csv(
            file_path, sep=cfg.datafmt_cfg['seperator'], encoding='utf-8',
            usecols=set(headers) - set(exclude_feats)
        )
        feat2type, inter_df = cls._convert_df_to_std_fmt(inter_df)
        if 'timestamp' in feat2type:
            inter_df = inter_df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

        # 1.重命名df、df_Q中exer_id字段为exer_id_old，将df_Q中exer_seq转换为集合，变换成新的exer_id
        # 2.每个df与df_Q left merge。删除每个df的旧exer_id字段
        feat2type_, df_Q = cls.read_Q_matrix(cfg)
        feat2type.update(feat2type_)
        
        df_Q.rename(columns={"exer_id": "exer_id_old"}, inplace=True)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].apply(lambda x: "_".join(list(map(lambda i: str(i), set(x)))))
        lbe = LabelEncoder()
        df_Q['cpt_seq'] = lbe.fit_transform(df_Q['cpt_seq'])
        df_Q.rename(columns={"cpt_seq": "exer_id"}, inplace=True)

        inter_df.rename(columns={'exer_id': 'exer_id_old'}, inplace=True)
        inter_df = inter_df.merge(df_Q, on='exer_id_old', how='left').reset_index(drop=True)
        inter_df.drop(labels=['exer_id_old'], axis=1, inplace=True)
        return inter_df

    @classmethod
    def read_Q_matrix(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datafmt_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_fmt(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q
