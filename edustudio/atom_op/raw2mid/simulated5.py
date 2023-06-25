import os
import re
import glob
from .raw2mid import BaseRaw2Mid
import pandas as pd
r"""
R2M_Simulated5
#####################################
Simulated5 dataset preprocess
"""


class R2M_Simulated5(BaseRaw2Mid):
    """R2M_Simulated5 is to preprocess Simulated5 dataset"""
    def process(self):
        super().process()

        def process_inter(path):
            # 读取数据，获得label
            df_inter = pd.read_csv(path, encoding='utf-8').reset_index()
            del df_inter['50']
            df_inter.columns = range(len(df_inter.columns))

            # 插入一行使其成为3的整数倍
            df_inter.index = df_inter.index + 1
            df_inter.loc[0] = 50
            df_inter = df_inter.reset_index(drop=True)

            # 对数据进行切片
            df_inter = df_inter.iloc[1::3]
            df_inter = df_inter.reset_index(drop=True)
            df_inter.insert(0, 'stu_id:token', range(len(df_inter)))

            # 拆成交互数据
            df_inter = df_inter.melt(id_vars=['stu_id:token'], var_name='exer_id:token', value_name='label:float')
            df_inter.sort_values(by=['stu_id:token', 'exer_id:token'], inplace=True)
            return df_inter

        def process_exer(path):
            # 读取文本文件，提取所需信息
            path = f'{self.rawpath}/cluster_info_0'
            with open(path, 'r') as file:
                content = file.read().replace('\n', '').replace('\t', '')
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, content)
            results = [match.split(', ') for match in matches][1:]
            df_exer = pd.DataFrame([(exer, cpt) for cpt, exers in enumerate(results) for exer in exers],
                                   columns=['exer_id:token', 'cpt_id:token'])
            df_exer['exer_id:token'] = df_exer['exer_id:token'].astype(int) - 1
            df_exer = df_exer.sort_values(by='exer_id:token')
            return df_exer

        # 处理后数据保存的文件夹路径
        output_folder = f"{self.midpath}"

        # 文件名模式
        file_pattern_inter = f"{self.rawpath}/naive_c5_q50_s4000_v*_train.csv"
        file_path_inters = glob.glob(file_pattern_inter)

        for file_path_inter in file_path_inters:
            processed_data = process_inter(file_path_inter)
            # 获取文件名
            file_name = f'{self.dt}_' + os.path.basename(file_path_inter)
            # 构建处理后数据的保存路径
            output_path = os.path.join(output_folder, file_name)
            # 生成新的文件名
            new_filename = output_path.replace(".csv", ".inter.csv")
            processed_data.to_csv(new_filename, index=False, encoding='utf-8')

        # 文件名模式
        file_pattern_exer = f"{self.rawpath}/cluster_info_*"
        file_path_exers = glob.glob(file_pattern_exer)

        for file_path_exer in file_path_exers:
            processed_data = process_exer(file_path_exer)
            file_name = f'{self.dt}_' + os.path.basename(file_path_exer)
            output_path = os.path.join(output_folder, file_name)
            new_file_name = output_path + '.exer.csv'
            processed_data.to_csv(new_file_name, index=False, encoding='utf-8')
