from ..common.base_mid2cache import BaseMid2Cache
import numpy as np
from edustudio.datatpl.utils import PadSeqUtil


class M2C_GenKCSeq(BaseMid2Cache):
    """Generate Knowledge Component Sequence
    """
    default_cfg = {
        'cpt_seq_window_size': -1,
    }

    def process(self, **kwargs):
        df_exer = kwargs['df_exer']
        tmp_df_Q = df_exer.set_index('exer_id:token')
        exer_count = kwargs['dt_info']['exer_count']
        
        cpt_seq_unpadding = [
            (tmp_df_Q.loc[exer_id].tolist()[0] if exer_id in tmp_df_Q.index else []) for exer_id in range(exer_count)
        ]
        cpt_seq_padding, _, cpt_seq_mask = PadSeqUtil.pad_sequence(
            cpt_seq_unpadding, maxlen=self.m2c_cfg['cpt_seq_window_size'], return_mask=True
        )

        kwargs['cpt_seq_padding'] = cpt_seq_padding
        kwargs['cpt_seq_mask'] = cpt_seq_mask
        return kwargs
