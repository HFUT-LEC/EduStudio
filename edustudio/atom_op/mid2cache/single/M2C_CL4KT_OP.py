from ..KT import M2C_BuildSeqInterFeats
import torch 
import pandas as pd
import numpy as np
from edustudio.datatpl.utils import PadSeqUtil
from collections import defaultdict


class M2C_CL4KT_OP(M2C_BuildSeqInterFeats):
    default_cfg = {
        'seed': 2023,
        'divide_by': 'stu',
        'window_size': 200,
        "divide_scale_list": [7,1,2],
        'sequence_truncation': '', # option: recent、history

    }
    def __init__(self, m2c_cfg, n_folds, is_dataset_divided) -> None:
        super().__init__(m2c_cfg, n_folds, is_dataset_divided)
    
    def process(self, **kwargs):
        kwargs = super().process(**kwargs)
        kwargs = self.compute_cpt2difflevel(**kwargs)

        return kwargs

    def compute_cpt2difflevel(self, **kwargs):
        cpt_correct = defaultdict(int)
        cpt_count = defaultdict(int)
        for i, (c_list, r_list) in enumerate(zip(kwargs['df_train_folds'][0]['cpt_unfold_seq:token_seq'], kwargs['df_train_folds'][0]['label_seq:float_seq'])):
            for c, r in zip(c_list[kwargs['df_train_folds'][0]['mask_seq:token_seq'][i] == 1], r_list[kwargs['df_train_folds'][0]['mask_seq:token_seq'][i] == 1]):
                cpt_correct[c] += r
                cpt_count[c] += 1
        cpt_diff = {c: cpt_correct[c] / float(cpt_count[c]) for c in cpt_correct}  # cpt difficult
        ordered_cpts = [item[0] for item in sorted(cpt_diff.items(), key=lambda x: x[1])]
        easier_cpts, harder_cpts = defaultdict(int), defaultdict(int)
        for index, cpt in enumerate(ordered_cpts):  
            if index == 0:
                easier_cpts[cpt] = ordered_cpts[index + 1]
                harder_cpts[cpt] = cpt
            elif index == len(ordered_cpts) - 1:
                easier_cpts[cpt] = cpt
                harder_cpts[cpt] = ordered_cpts[index - 1]
            else:
                easier_cpts[cpt] = ordered_cpts[index + 1]
                harder_cpts[cpt] = ordered_cpts[index - 1]

        kwargs['easier_cpts'] = easier_cpts
        kwargs['harder_cpts'] = harder_cpts

        return kwargs

    def construct_df2dict(self, df: pd.DataFrame):
        if df is None: return None
     
        if self.m2c_cfg['sequence_truncation'] == 'recent':
            tmp_df = df[['stu_id:token', 'exer_id:token', 'cpt_unfold:token', 'label:float']].groupby('stu_id:token').agg(
                lambda x: list(x)[-self.m2c_cfg['window_size']:]
            ).reset_index()

            exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(  # mask_seq corresponding to attention_mask
                tmp_df['exer_id:token'].to_list(), return_idx=True, return_mask=True,
                maxlen=self.m2c_cfg['window_size'], padding='pre'
            )
            cpt_unfold_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['cpt_unfold:token'].to_list(),
                maxlen=self.m2c_cfg['window_size'], padding='pre'
            )
            label_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['label:float'].to_list(), dtype=np.float32,
                maxlen=self.m2c_cfg['window_size'], padding='pre'
            )
            
        else:
            tmp_df = df[['stu_id:token', 'exer_id:token', 'cpt_unfold:token', 'label:float']].groupby('stu_id:token').agg(
                lambda x: list(x)[:self.m2c_cfg['window_size']]
            ).reset_index()
            exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(  # mask_seq corresponding to attention_mask
                tmp_df['exer_id:token'].to_list(), return_idx=True, return_mask=True,
                maxlen=self.m2c_cfg['window_size'],
            )
            cpt_unfold_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['cpt_unfold:token'].to_list(),
                maxlen=self.m2c_cfg['window_size'],
            )
            label_seq, _, _ = PadSeqUtil.pad_sequence(
                tmp_df['label:float'].to_list(), dtype=np.float32,
                maxlen=self.m2c_cfg['window_size'],
            )


        stu_id = tmp_df['stu_id:token'].to_numpy()[idx]

        ret_dict = {
            'stu_id:token': stu_id,
            'exer_seq:token_seq': exer_seq,
            'cpt_unfold_seq:token_seq': cpt_unfold_seq,
            'label_seq:float_seq': label_seq,
            'mask_seq:token_seq': mask_seq
        }

        return ret_dict
    
    def set_dt_info(self, dt_info, **kwargs):
        super().set_dt_info(dt_info, **kwargs)
        dt_info['train_easier_cpts'] = kwargs['easier_cpts']
        dt_info['train_harder_cpts'] = kwargs['easier_cpts']
