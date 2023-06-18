import numpy as np

from .kt_inter_datafmt import KTInterDataFmt
from itertools import chain
import torch
import pandas as pd
from scipy import sparse
from ..utils.pad_seq_util import PadSeqUtil


class RKTDataFmt(KTInterDataFmt):
    default_cfg = {

    }

    def __init__(self,
                 cfg,
                 train_dict,
                 val_dict,
                 test_dict,
                 feat2type,
                 **kwargs
                ):
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type)
        self.get_corr_data()


    def get_corr_data(self):
        file_path = f'{self.cfg.frame_cfg.data_folder_path}/pro_pro_sparse.npz'
        # pro_pro_dense = np.zeros((self.n_item, self.n_item))
        pro_pro_sparse = sparse.load_npz(file_path)
        pro_pro_coo = pro_pro_sparse.tocoo()
        # print(pro_skill_csr)
        self.pro_pro_dense = pro_pro_coo.toarray()

    def get_extra_data(self):
        return {
            "pro_pro_dense": self.pro_pro_dense
        }

    @classmethod
    def construct_df2dict(cls, cfg, df: pd.DataFrame, maxlen=None):
        if df is None: return None

        tmp_df = df[['stu_id', 'exer_id', 'label', 'timestamp']].groupby('stu_id').agg(
            lambda x: list(x)
        ).reset_index()

        exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(
            tmp_df['exer_id'].to_list(), return_idx=True, return_mask=True,
            maxlen=maxlen
        )
        label_seq, _, _ = PadSeqUtil.pad_sequence(
            tmp_df['label'].to_list(), dtype=np.float32,
            maxlen=maxlen
        )

        time_seq, _, _, = PadSeqUtil.pad_sequence(
            tmp_df['timestamp'].to_list(), dtype=np.int64,
            maxlen=maxlen
        )

        stu_id = tmp_df['stu_id'].to_numpy()[idx]

        item_inputs = np.hstack((np.zeros((exer_seq.shape[0], 1)), exer_seq[:, :-1]))
        label_inputs = np.hstack((np.zeros((exer_seq.shape[0], 1)), label_seq[:, :-1]))

        return {
            'stu_id': stu_id,
            'exer_seq': exer_seq,
            'label_seq': label_seq,
            'time_seq': time_seq,
            'item_inputs': item_inputs,
            'label_inputs': label_inputs,
            'mask_seq': mask_seq
        }
