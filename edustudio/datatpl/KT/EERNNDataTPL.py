
import numpy as np
from ..common import EduDataTPL


class EERNNDataTPL(EduDataTPL):
    default_cfg = {
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_ReMapId', 'M2C_BuildSeqInterFeats','M2C_RandomDataSplit4KT', 'M2C_EERNN_OP'],
    }
    
    def get_extra_data(self, **kwargs):
        super_dic = super().get_extra_data(**kwargs)
        super_dic['w2v_word_emb'] = self.w2v_word_emb
        super_dic['exer_content'] = self.content_mat
        return super_dic
    
    def set_info_for_fold(self, fold_id):
        dt_info = self.datatpl_cfg['dt_info']
        dt_info['word_count'] = len(self.word_emb_dict_list[fold_id])

        self.w2v_word_emb = np.vstack(
            [self.word_emb_dict_list[fold_id][k] for k in range(self.datatpl_cfg['dt_info']['word_count'])]
        )

        self.content_mat = self.content_mat_list[fold_id]

    def save_cache(self):
        super().save_cache()
        fph1 = f"{self.cache_folder_path}/word_emb_dict_list.pkl"
        fph2 = f"{self.cache_folder_path}/content_mat_list.pkl"
        self.save_pickle(fph1, self.word_emb_dict_list)
        self.save_pickle(fph2, self.content_mat_list)

    def load_cache(self):
        super().load_cache()
        fph1 = f"{self.cache_folder_path}/word_emb_dict_list.pkl"
        fph2 = f"{self.cache_folder_path}/content_mat_list.pkl"
        self.word_emb_dict_list = self.load_pickle(fph1)
        self.content_mat_list = self.load_pickle(fph2)
