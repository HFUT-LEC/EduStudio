from ..common import EduDataTPL


class CDGKDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat', 'M2C_CDGK_OP'],
    }

    # @classmethod
    # def load_data(cls, cfg): # 只在middata存在时调用
    #     kwargs = super().load_data(cfg)
    #     if cfg.datatpl_cfg['has_cpt2group_file'] is True:
    #         new_kwargs = cls.load_cpt2group(cfg)
    #         for df in new_kwargs.values(): 
    #             if df is not None:
    #                 cls._preprocess_feat(df) # 类型转换
    #         kwargs.update(new_kwargs)
    #     else:
    #         kwargs['df_cpt2group'] = None
    #     return kwargs
    

    # @classmethod
    # def load_cpt2group(cls, cfg):
    #     cpt2group_fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.cpt2group.csv'
    #     sep = cfg.datatpl_cfg['seperator']
    #     df_cpt2group = cls._load_atomic_csv(cpt2group_fph, sep=sep)
    #     return {"df_cpt2group": df_cpt2group}
