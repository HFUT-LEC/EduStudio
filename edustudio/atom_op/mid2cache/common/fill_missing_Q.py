from .base_mid2cache import BaseMid2Cache
import numpy as np
import pandas as pd
from itertools import chain
import torch
from edustudio.utils.common import set_same_seeds, tensor2npy
from tqdm import tqdm

class M2C_FillMissingQ(BaseMid2Cache):
    default_cfg = {
        'Q_fill_type': "None",
        'params_topk': 5, 
        'params_votek': 2,
    }

    def __init__(self, m2c_cfg, cfg) -> None:
        self.logger = cfg.logger
        self.m2c_cfg = m2c_cfg
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.datatpl_cfg.get(cls.__name__), cfg)

    def process(self, **kwargs):
        dt_info = kwargs['dt_info']
        self.user_count = dt_info['stu_count']
        self.item_count = dt_info['exer_count']
        self.cpt_count = dt_info['cpt_count']
        self.df_Q = kwargs['df_exer'][['exer_id:token', 'cpt_seq:token_seq']]

        Q_mat = kwargs['Q_mat']
        missing_Q_mat = kwargs['missing_Q_mat']

        self.filling_Q_mat_list = []
        for df_train in kwargs['df_train_folds']:
            if (missing_Q_mat.sum(dim=1) == 0).sum() > 0:
                if self.m2c_cfg['Q_fill_type'] == "sim_dist_for_by_exer":
                    fill_df_Q = self.fill_df_Q_by_sim_dist(
                        df_train, kwargs['missing_df_Q'], 
                        params_topk=self.m2c_cfg['params_topk'], 
                        params_votek=self.m2c_cfg['params_votek']
                    )
                    fill_Q_mat = self.get_Q_mat_from_df_arr(fill_df_Q, self.item_count, self.cpt_count)
                    self.filling_Q_mat_list.append(fill_Q_mat)
                elif self.m2c_cfg['Q_fill_type'] == "None":
                    self.filling_Q_mat_list.append(missing_Q_mat)
                else:
                    raise ValueError(f"unknown Q_fill_type: {self.m2c_cfg['Q_fill_type']}")
            else:
                self.filling_Q_mat_list.append(Q_mat)

        kwargs['filling_Q_mat_list'] = self.filling_Q_mat_list
        return kwargs

    def get_Q_mat_from_df_arr(self, df_Q_arr, item_count, cpt_count):
        Q_mat = np.zeros((item_count, cpt_count), dtype=np.int64)
        for _, item in df_Q_arr.iterrows(): Q_mat[item['exer_id:token'], item['cpt_seq:token_seq']] = 1
        return Q_mat

    def fill_df_Q_by_sim_dist(self, df_interaction, df_Q_left, params_topk=5, params_votek=2):
        preserved_exers = df_Q_left['exer_id:token'].to_numpy()
        interact_mat = torch.zeros((self.user_count, self.item_count), dtype=torch.int8).to(self.cfg.traintpl_cfg['device'])
        idx = df_interaction[df_interaction['label:float'] == 1][['stu_id:token','exer_id:token']].to_numpy()
        interact_mat[idx[:,0], idx[:,1]] = 1  
        idx = df_interaction[df_interaction['label:float'] != 1][['stu_id:token','exer_id:token']].to_numpy()
        interact_mat[idx[:,0], idx[:,1]] = -1 

        interact_mat = interact_mat.T

        sim_mat = torch.zeros((self.item_count, self.item_count))
        missing_iids = np.array(list(set(np.arange(self.item_count)) - set(preserved_exers)))
        for iid in tqdm(missing_iids, desc="[FILL_Q_MAT] compute sim_mat", ncols=self.cfg.frame_cfg['TQDM_NCOLS']):
            temp = interact_mat[iid] != 0
            same_mat =  interact_mat[iid] == interact_mat
            bool_mat = (temp) & (interact_mat != 0) 
            same_mat[~bool_mat] = False
            sim_mat[iid] = same_mat.sum(dim=1) / (temp).sum()
            sim_mat[iid, bool_mat.sum(dim=1) == 0] = 0.0
            sim_mat[iid, iid] = -1.0
            sim_mat[iid, missing_iids] = -1.0

        assert torch.isnan(sim_mat).sum() == 0

        _, topk_mat_idx = torch.topk(sim_mat, dim=1, k=params_topk, largest=True, sorted=True)
        topk_mat_idx = tensor2npy(topk_mat_idx)

        index_df_Q = df_Q_left.set_index('exer_id:token')
        missing_iid_fill_cpts = {}
        for iid in tqdm(missing_iids, desc="[FILL_Q_MAT] fill process", ncols=self.cfg.frame_cfg['TQDM_NCOLS']):
            count_dict = dict(zip(*np.unique(
                list(chain(*[index_df_Q.loc[iid2]['cpt_seq:token_seq'] for iid2 in topk_mat_idx[iid] if iid2 in preserved_exers])),
                return_counts=True,
            )))
            count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
            missing_iid_fill_cpts[iid] = [i[0] for i in count_dict[0:params_votek]]

        missing_fill_df_Q = pd.DataFrame(
            {'exer_id:token': list(missing_iid_fill_cpts.keys()),'cpt_seq:token_seq':list(missing_iid_fill_cpts.values())}
        )
        final_df_Q = pd.concat([df_Q_left, missing_fill_df_Q], axis=0, ignore_index=True)

        hit_ratio = 0
        t_Q = self.df_Q.set_index('exer_id:token')
        for iid in missing_iid_fill_cpts:
            if len(set(t_Q.loc[iid]['cpt_seq:token_seq']) & set(missing_iid_fill_cpts[iid])) > 0:
                hit_ratio += 1
        hit_ratio = hit_ratio / len(missing_iid_fill_cpts)

        self.logger.info(f"[FILL_Q] Hit_ratio={hit_ratio}")

        return final_df_Q
