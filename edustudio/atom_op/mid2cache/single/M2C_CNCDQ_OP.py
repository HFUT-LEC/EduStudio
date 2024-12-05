from ..common import BaseMid2Cache
import torch

class M2C_CNCDQ_OP(BaseMid2Cache):
    default_cfg = {}

    def process(self, **kwargs):
        df_exer = kwargs['df_exer']
        exer_count = kwargs['dt_info']['exer_count']
        cpt_count = kwargs['dt_info']['cpt_count']
        kwargs['Q_mask_mat'], kwargs['knowledge_pairs'] = self._get_knowledge_pairs(
            df_Q_arr=df_exer, exer_count=exer_count, cpt_count=cpt_count
        )
        return kwargs

    def _get_knowledge_pairs(self, df_Q_arr, exer_count, cpt_count):
        Q_mask_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
        knowledge_pairs = []
        kn_tags = []
        kn_topks = []
        for _, item in df_Q_arr.iterrows():
            # kn_tags.append(item['cpt_seq'])
            # kn_topks.append(item['cpt_pre_seq'])
            kn_tags = item['cpt_seq']
            kn_topks = item['cpt_pre_seq']
            knowledge_pairs.append((kn_tags, kn_topks))
            for cpt_id in item['cpt_seq']:
                Q_mask_mat[item['exer_id'], cpt_id-1] = 1
            for cpt_id in item['cpt_pre_seq']:
                Q_mask_mat[item['exer_id'], cpt_id-1] = 1
        return Q_mask_mat, knowledge_pairs
