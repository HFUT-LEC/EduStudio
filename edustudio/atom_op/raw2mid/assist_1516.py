import pandas as pd
from .raw2mid import BaseRaw2Mid

r"""
R2M_ASSIST_1516
########################
"""


class R2M_ASSIST_1516(BaseRaw2Mid):
    """R2M_ASSIST_1516 is a class used to handle the ASSISTment 2015-2016 dataset."""

    def process(self):
        super().process()
        pd.set_option("mode.chained_assignment", None)  # ignore warning
        # 读取原始数据，查看其属性
        raw_data = pd.read_csv(f"{self.rawpath}/2015_100_skill_builders_main_problems.csv", encoding='utf-8')

        # 获取交互信息
        # 对log_id进行排序，确定order序列
        inter = pd.DataFrame.copy(raw_data).sort_values(by='log_id', ascending=True)
        inter['order'] = range(len(inter))
        inter['label'] = inter['correct']
        df_inter = inter.rename(columns={'sequence_id': 'exer_id', 'user_id': 'stu_id'}).reindex(
            columns=['stu_id', 'exer_id', 'label', 'order', ]).rename(
            columns={'stu_id': 'stu_id:token', 'exer_id': 'exer_id:token', 'label': 'label:float',
                     'order': 'order_id:token'})

        # 获取学生信息
        stu = pd.DataFrame(set(raw_data['user_id']), columns=['stu_id', ])
        # stu['classId'] = None
        # stu['gender'] = None
        df_stu = stu.sort_values(by='stu_id', ascending=True)

        # 获取习题信息
        exer = pd.DataFrame(set(raw_data['sequence_id']), columns=['exer_id'])
        # exer['cpt_seq'] = None
        # exer['assignment_id'] = None
        df_exer = exer.sort_values(by='exer_id', ascending=True)

        # 此处将数据保存到`self.midpath`中

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        # df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        pd.set_option("mode.chained_assignment", "warn")  # ignore warning
        return
