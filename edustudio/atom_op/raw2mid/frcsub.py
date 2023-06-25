from .raw2mid import BaseRaw2Mid
import pandas as pd
pd.set_option("mode.chained_assignment", None) # ingore warning

r"""
R2M_FrcSub
#####################################
FrcSub dataset preprocess
"""

class R2M_FrcSub(BaseRaw2Mid):
    """R2M_FrcSub is to preprocess FrcSub dataset"""
    def process(self):
        super().process()
        # # Preprocess
        # 
        # 此处对数据集进行处理

        # 读取文本文件转换为 dataframe
        df_inter = pd.read_csv(f"{self.rawpath}/data.txt", sep='\t', names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                                                                        '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']) 
        df_inter.insert(0, 'stu_id:token', range(len(df_inter)))
        df_exer = pd.read_csv(f"{self.rawpath}/q.txt", sep='\t', names=['0', '1', '2', '3', '4', '5', '6', '7'])
        df_exer.insert(0, 'exer_id:token', range(len(df_exer)))
        # print(df_inter)
        # print(df_exer)

        # 统计知识点个数
        cpt_count = df_exer.shape[1]

        # 处理 df_inter 拆成很多行
        df_inter = df_inter.melt(id_vars=['stu_id:token'], var_name='exer_id:token', value_name='label:float')
        df_inter['exer_id:token'] = df_inter['exer_id:token'].astype(int)
        df_inter .sort_values(by = ['stu_id:token','exer_id:token'], inplace=True)
        # print(df_inter)

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
        # print(df_exer)

        # # Stat dataset
        # 
        # 此处统计数据集，保存到cfg对象中

        # cfg.stu_count = len(df_inter['stu_id:token'].unique())
        # cfg.exer_count = len(df_exer)
        # cfg.cpt_count = cpt_count
        # cfg.interaction_count = len(df_inter)
        # cfg

        # # Save MidData
        # 
        # 此处将数据保存到`cfg.MIDDATA_PATH`中

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
