import pandas as pd
from .raw2mid import BaseRaw2Mid

r"""
R2M_Junyi_Area_Topic_As_Cpt
########################
"""


class R2M_Junyi_AreaTopicAsCpt(BaseRaw2Mid):
    """R2M_Junyi_Area_Topic_As_Cpt is a class used to handle the Junyi dataset, where we consider the exercise's KC(Area) and KC(Topic) as the basis for constructing the cpt_seq (concept sequence)."""

    def process(self):
        super().process()
        pd.set_option("mode.chained_assignment", None)  # ignore warning
        # 读取数据集，并展示相关数据
        raw_data = pd.read_table(f"{self.rawpath}/junyi_ProblemLog_for_PSLC.txt", sep='\t', encoding='utf-8',
                                 low_memory=True)
        data = raw_data.head()
        # 去除提示的交互
        data = data.drop(data[data.Outcome == 'HINT'].index)
        data = data[
            ['Anon Student Id', 'Time', 'Level (Section)', 'Problem Name', 'Problem Start Time', 'Outcome',
             'KC (Exercise)',
             'KC (Topic)', 'KC (Area)']]
        data = data.sort_values(by='Time', ascending=True)
        data['order:token'] = range(len(data))
        data['cost_time:float'] = data['Time'] - data['Problem Start Time']

        inter = data.rename(
            columns={'Anon Student Id': 'stu_id:token', 'Problem Name': 'exer_name:token', 'Outcome': 'label:float',
                     'Problem Start Time': 'start_timestamp:float', }) \
            .reindex(
            columns=['stu_id:token', 'exer_name:token', 'label:float', 'start_timestamp:float', 'cost_time:float',
                     'order:token'])

        # 定义一个name_to_id的函数，返回值为两个dict,一个是name2id，一个是id2name
        # 主要用于将唯一标识的字符串生成自增的id
        def name_to_id(l: list()):
            if l is None:
                return None, None
            l = list(l)
            name2id = dict(zip(l, range(len(l))))
            id2name = dict(zip(range(len(l)), l))
            return name2id, id2name

        # 进行exer_name 转 exer_id
        unique_exer = list(inter['exer_name:token'].unique())
        exer2id, id2exer = name_to_id(unique_exer)
        inter['exer_id:token'] = inter['exer_name:token'].map(exer2id)
        # 确定label
        str2label = {'CORRECT': 1, 'INCORRECT': 0}
        inter['label:float'] = inter['label:float'].map(str2label)
        df_inter = inter[
            ['stu_id:token', 'exer_id:token', 'label:float', 'start_timestamp:float', 'cost_time:float', 'order:token']]

        # 处理用户信息
        df_stu = data['Anon Student Id'].unique()
        # df_stu['class_id'] = None
        # df_stu['gender'] = None

        # 处理习题信息
        exer = data[['Problem Name', 'Level (Section)', 'KC (Exercise)', 'KC (Area)', 'KC (Topic)']].copy()
        # 处理习题，先去重
        exer = exer.drop_duplicates(subset=['Problem Name'], keep='first')
        exer['exer_id'] = exer['Problem Name'].map(exer2id)
        # 处理assignment name to id的映射
        unique_assignment = list(exer['Level (Section)'].unique())
        assignment2id, id2assignment = name_to_id(unique_assignment)
        exer['assignment_id'] = exer['Level (Section)'].map(assignment2id)
        # 获取KCs
        exer['KC (Area)'] = exer['KC (Area)'] + '&Area'  # 标明知识点来自Area
        exer['KC (Topic)'] = exer['KC (Topic)'] + '&Topic'  # 标明知识点来自Topic
        unique_kc_area = set(exer['KC (Area)'])
        unique_kc_topic = set(exer['KC (Topic)'])
        # 基于Area和Topic字段生成所有KC,及kc2id,id2kc
        unique_kc = list(unique_kc_area | unique_kc_topic)
        kc2id, id2kc = name_to_id(unique_kc)
        exer['cpt_seq_area'] = exer['KC (Area)'].map(lambda x: kc2id[x])
        exer['cpt_seq_topic'] = exer['KC (Topic)'].map(lambda x: kc2id[x])
        exer['cpt_seq'] = '[' + exer['cpt_seq_area'].map(str) + ',' + exer['cpt_seq_topic'].map(str) + ']'

        df_exer = exer.rename(columns={
            'exer_id': 'exer_id:token', 'assignment_id': 'assignment_id:token_seq',
            'cpt_seq': 'cpt_seq:token_seq',
            'cpt_seq_area': 'cpt_seq_level_1:token_seq', 'cpt_seq_topic': 'cpt_seq_level_2:token_seq'
            # area - level1;    topic - level2
        }).reindex(
            columns=['exer_id:token', 'assignment_id:token_seq', 'cpt_seq:token_seq', 'cpt_seq_level_1:token_seq',
                     'cpt_seq_level_2:token_seq'])

        # 处理cpt_tree信息
        # topic ∈ area
        topic_area = exer.drop_duplicates(subset=['cpt_seq_topic'], inplace=False)[['cpt_seq_topic', 'cpt_seq_area']]
        # cpt_id:token    parent_id:token    level:float
        cpt_data = []
        # topic, area, level = 2
        topic_area.apply(lambda x: cpt_data.append([x['cpt_seq_topic'], x['cpt_seq_area'], 2]), axis=1)
        area = topic_area.drop_duplicates(subset=['cpt_seq_area'], inplace=False)[['cpt_seq_area']]
        # area, none, level = 1
        area.apply(lambda x: cpt_data.append([x['cpt_seq_area'], None, 1]), axis=1)
        df_cpt = pd.DataFrame(columns=['cpt_id:token', 'parent_id:token', 'level:float'], data=cpt_data)
        df_cpt.sort_values(by='level:float', ascending=True, inplace=True)

        # 此处将数据保存到`self.midpath`中

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        df_cpt.to_csv(f"{self.midpath}/{self.dt}.cpt_tree.csv", index=False, encoding='utf-8')
        pd.set_option("mode.chained_assignment", "warn")  # ignore warning
        return
