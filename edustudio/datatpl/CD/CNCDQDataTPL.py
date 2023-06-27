from ..common import EduDataTPL
import torch


class CNCDQDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    }

    def get_extra_data(self, **kwargs):
        dic = super().get_extra_data(**kwargs)
        _, _, dic['knowledge_pairs'] = self._get_knowledge_pairs(
            self.df_exer, self.datatpl_cfg['dt_info']['exer_count'], self.datatpl_cfg['dt_info']['cpt_count']
        )
        return dic

    def _get_knowledge_pairs(self, df_Q_arr, exer_count, cpt_count):
        Q_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
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
                Q_mat[item['exer_id'], cpt_id-1] = 1
                Q_mask_mat[item['exer_id'], cpt_id-1] = 1
            for cpt_id in item['cpt_pre_seq']:
                Q_mask_mat[item['exer_id'], cpt_id-1] = 1
        return Q_mat, Q_mask_mat, knowledge_pairs
