from .raw2mid import BaseRaw2Mid
import pandas as pd

r"""
R2M_Math2
#####################################
Math2 dataset preprocess
"""


class R2M_Math2(BaseRaw2Mid):
    """R2M_Math2 is to preprocess Math2 dataset"""
    def process(self):
        super().process()

        # 读取文本文件转换为 dataframe
        df_inter = pd.read_csv(f"{self.rawpath}/data.txt", sep='\t', names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                                                        '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']) 
        df_inter.insert(0, 'stu_id:token', range(len(df_inter)))
        df_exer = pd.read_csv(f"{self.rawpath}/q.txt", sep='\t', names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                                                                           '11', '12', '13', '14', '15'])
        df_exer.insert(0, 'exer_id:token', range(len(df_exer)))

        # 处理 df_inter 拆成很多行
        df_inter = df_inter.melt(id_vars=['stu_id:token'], var_name='exer_id:token', value_name='label:float')
        df_inter['exer_id:token'] = df_inter['exer_id:token'].astype(int)
        df_inter .sort_values(by = ['stu_id:token','exer_id:token'], inplace=True)

        # 处理 df_exer 先拆成很多行，再合并
        # 拆成很多行
        df_exer = df_exer.melt(id_vars=['exer_id:token'], var_name='cpt_seq:token_seq', value_name='value')
        df_exer = df_exer[df_exer['value'] == 1]
        del df_exer['value']
        
        # 合并 cpt_seq:token_seq
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].astype(int)
        df_exer.sort_values(by='cpt_seq:token_seq', inplace=True)
        df_exer['exer_id:token'] = df_exer['exer_id:token'].astype(str)
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].astype(str)
        df_exer  = df_exer.groupby('exer_id:token')['cpt_seq:token_seq'].agg(','.join).reset_index()

        # 按 exer_id:token 进行排序
        df_exer['exer_id:token'] = df_exer['exer_id:token'].astype(int)
        df_exer.sort_values(by='exer_id:token', inplace=True)

        # 保存数据
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')