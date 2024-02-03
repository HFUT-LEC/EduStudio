from ..common import EduDataTPL
import torch
import numpy as np
import pandas as pd


class DKTDSCDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ["M2C_KCAsExer", 'M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats','M2C_RandomDataSplit4KT', "M2C_DKTDSC_OP"],
    }

    def __getitem__(self, index):
        dic = super().__getitem__(index)
        # dic['cluster'] = self.cluster[(int(dic['stu_id']), int(dic['seg_seq']))]
        step = len(dic['seg_seq'])
        stu_id = dic['stu_id'].repeat(step)
        # cluster_df = pd.DataFrame([[list(each)] for each in torch.cat((stu_id.unsqueeze(1), dic['seg_seq'].unsqueeze(1)), dim=1).numpy()], columns=['stu_seg_id'])
        # result = pd.merge(cluster_df, self.cluster, on = ['stu_seg_id']).reset_index(drop=True)
        # cluster_id_tensor = torch.Tensor(result['cluster_id'].values)
        # dic['cluster'] = cluster_id_tensor
        dic['cluster'] = np.ones_like(dic['exer_seq'])
        for i in range(step):
            try:
                dic['cluster'][i] = self.cluster.get((int(stu_id[i]), int(dic['seg_seq'][i])))
            except:
                dic['cluster'][i] = 0
        dic['cluster'] = torch.from_numpy(dic['cluster'])
        return dic


    def set_info_for_fold(self, fold_id):
        super().set_info_for_fold(fold_id)
        self.cluster = self.final_kwargs['cluster_list'][fold_id]
