import pandas as pd
import random
import os
from .raw2mid import BaseRaw2Mid

r"""
R2M_EdNet_KT1
########################
"""


class R2M_EdNet_KT1(BaseRaw2Mid):
    """R2M_EdNet_KT1 is a class used to handle the EdNet-KT1 dataset."""

    def process(self):
        super().process()
        pd.set_option("mode.chained_assignment", None)  # ignore warning
        pd.set_option("mode.chained_assignment", None)  # ignore warning
        # 读取数据集，并展示相关数据
        # 数据集包括784,309个学生，每个学生有一个交互的csv文件，这里我们进行抽样其中的5000个
        # 抽样其中的5000个
        all_files = pd.Series(os.listdir(f"{self.rawpath}")).to_list()
        all_files.remove("contents")  # 除去contents文件夹
        random.seed(2)
        files = random.sample(all_files, 5000)  # 采样个数5000个
        all_data = []
        for file_name in files:
            data = pd.read_csv(f"{self.rawpath}" + '\\' + file_name, encoding="utf-8")
            data['stu_id'] = int(file_name[:-4][1:])
            # 先把每个用户的数据暂存到列表[]中， 后一次性转化为DataFrame
            all_data.append(data)
        df = pd.concat(all_data)
        # 确定order字段
        df = df.sort_values(by='timestamp', ascending=True)
        df['order'] = range(len(df))

        # 读取question.csv,判断用户作答情况，是否正确
        question = pd.read_csv(os.path.join(f"{self.rawpath}", 'contents', 'questions.csv'))
        inter = df.merge(question, sort=False, how='left')
        inter = inter.dropna(subset=["stu_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
        inter['label'] = (inter['correct_answer'] == inter['user_answer']).apply(int)
        inter['exer_id'] = inter['question_id'].apply(lambda x: x[1:])

        # 交互信息
        df_inter = inter \
            .reindex(columns=['stu_id', 'exer_id', 'label', 'timestamp', 'elapsed_time', 'order']) \
            .rename(columns={'stu_id': 'stu_id:token', 'exer_id': 'exer_id:token', 'label': 'label:float',
                             'timestamp': 'start_timestamp:float', 'elapsed_time': 'cost_time:float',
                             'order': 'order_id:token'})

        # 处理用户信息
        df_stu = df_inter['stu_id:token'].copy().unique()

        # 处理习题信息
        exer = question.copy()
        exer['cpt_seq'] = exer['tags'].apply(lambda x: x.split(';')).apply(lambda x: list(map(int, x)))
        exer['exer_id'] = exer['question_id'].apply(lambda x: x[1:])
        # 存储所有的cpt，即知识点id
        kcs = set()
        for cpt in exer['cpt_seq']:
            kcs.update(cpt)
        df_exer = exer.reindex(columns=['exer_id', 'cpt_seq']).rename(columns={
            'exer_id': 'exer_id:token', 'cpt_seq': 'cpt_seq:token_seq'
        })
        # 此处将数据保存到`cfg.MIDDATA_PATH`中
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{cfg.MIDDATA_PATH}/{cfg.DT}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        pd.set_option("mode.chained_assignment", "warn")  # ignore warning
        return
