import pandas as pd
import time
from .raw2mid import BaseRaw2Mid

r"""
R2M_Bridge2Algebra_0607
########################
"""


class R2M_Bridge2Algebra_0607(BaseRaw2Mid):
    """R2M_Bridge2Algebra_0607 is a class used to handle the Bridge2Algebra 2006-2007 dataset."""

    def process(self):
        super().process()
        pd.set_option("mode.chained_assignment", None)  # ignore warning
        # 读取数据集，并展示相关数据
        raw_data_train = pd.read_table(f"{self.rawpath}/bridge_to_algebra_2006_2007_train.txt", encoding='utf-8',
                                       low_memory=False)
        raw_data_test = pd.read_table(f"{self.rawpath}/bridge_to_algebra_2006_2007_master.txt", encoding='utf-8',
                                      low_memory=False)
        raw_data = pd.concat([raw_data_train, raw_data_test], axis=0)
        raw_data.head()

        # 预处理信息
        # 1.去除空值
        data = raw_data.copy().dropna(subset=['Step Start Time'])
        # 2. 时间格式转换为时间戳, 单位是ms
        data['Step Start Time'] = data['Step Start Time'].apply(
            lambda x: int(time.mktime(time.strptime(str(x)[:-2], "%Y-%m-%d %H:%M:%S"))) * 1000)

        # 以'Anon Student Id', 'Problem Name'为分组依据，将学生处理每个习题的步骤合并到一起
        # 同时处理一些数据，包括assignment_name，cost_time，start_timestamp，correct
        def concat_func(x):
            return pd.Series({
                'assignment_name': x['Problem Hierarchy'].unique(),
                'cost_time': x['Step Duration (sec)'].sum(),
                'steps': ','.join(x['Step Name']),
                'start_timestamp': x['Step Start Time'].min(),
                'correct': x['Corrects'].min(),  # 我们认为只要学生步骤中有一步没有作对，那么整个题就不对
                'KC(SubSkills)': '--'.join(str(id) for id in x['KC(SubSkills)'])  # 遇到none怎么办？
            })

        raw_data = data.groupby(['Anon Student Id', 'Problem Name'], as_index=False).apply(concat_func)

        # 定义一个name_to_id的函数，返回值为两个dict,一个是name2id，一个是id2name
        # 主要用于将唯一标识的字符串生成自增的id
        def name_to_id(l: list()):
            if l is None:
                return None, None
            name2id = dict(zip(l, range(len(l))))
            id2name = dict(zip(range(len(l)), l))
            return name2id, id2name

        # 获取交互信息
        # 若用户正确次数超过一次，则认为用户已经正确做答
        def correct(corrects):
            if corrects >= 1:
                return 1
            return 0

        inter = pd.DataFrame.copy(raw_data)
        # 接下来就添加label、order、cost_time信息
        inter['label'] = inter['correct'].apply(lambda x: correct(x))
        # 根据时间戳确定学生做题的序列
        inter = inter.sort_values(by='start_timestamp', ascending=True)
        inter['order'] = range(len(inter))
        # cost_time 转化为以ms为单位
        inter['cost_time'] = inter['cost_time'].apply(lambda x: int(x * 1000))
        # 先重命名，方便后续操作
        df_inter = inter.rename(columns={'Anon Student Id': 'stu_name', 'Problem Name': 'exer_name'})

        # 处理name to id的映射，及最终df_inter的标准格式
        # 处理id映射，包括将字符类型的stu_name,exer_name映射到数字类型的stu_id和exer_id
        unique_stu = list(df_inter['stu_name'].copy().unique())
        unique_exer = list(df_inter['exer_name'].copy().unique())
        # 设置stu2id,和id2stu的dict及exer2id和id2exer的dict
        stu2id, id2stu = name_to_id(unique_stu)
        exer2id, id2exer = name_to_id(unique_exer)
        df_inter['stu_id'] = df_inter['stu_name'].apply(lambda x: stu2id[x])
        df_inter['exer_id'] = df_inter['exer_name'].apply(lambda x: exer2id[x])
        # 最终处理，重排列序
        df_inter = df_inter.reindex(
            columns=['stu_id', 'exer_id', 'label', 'start_timestamp', 'cost_time', 'order']).rename(
            columns={'stu_id': 'stu_id:token', 'exer_id': 'exer_id:token', 'label': 'label:float',
                     'start_timestamp': 'start_timestamp:float', 'cost_time': 'cost_time:float',
                     'order': 'order_id:token'})

        # 处理用户信息
        df_stu = df_inter['stu_id:token'].copy().unique()
        # df_stu['classId'] = None
        # df_stu['gender'] = None

        # 处理习题信息
        # kcs是一个set集合，包括整个训练集的所有cpt信息;ks为每个习题所关联的cpt_seq信息
        # cpt2id 是 cpt_name 到 cpt_id 的字典;id2cpt 是 cpt_id 到 cpt_name 的字典
        kcs = set()
        cpt2id = dict()
        id2cpt = dict()

        # 对KC(Default)进行预处理，将其转化cpt_seq_name序列集合
        def preprocess_cpt_seq(s):
            cpt_seq_name = set()
            if type(s) == str:
                kc = s.split("--")
                for i in kc:
                    if i[0] == '[':
                        i = [s.strip("- {}]") for s in i[12:].split(";")]
                        cpt_seq_name.update(i)
                    else:
                        cpt_seq_name.add(i)
            if 'nan' in cpt_seq_name:  # 去除空值nan
                cpt_seq_name.remove('nan')
            # cpt_seq_name.pop('nan')
            kcs.update(cpt_seq_name)
            if len(cpt_seq_name) == 0:
                cpt_seq_name = None
            return cpt_seq_name

        # 将cpt_seq_name的字符串集合转化为cpt_seq的数字列表
        def process_cpt_seq(cpt_seq_name: set()):
            if cpt_seq_name is None:
                return None
            cpt_seq = [cpt2id[cpt_name] for cpt_name in cpt_seq_name]
            cpt_seq.sort()
            return cpt_seq

        # 对exercise的处理,主要将KC(Default)处理成cpt_seq_name序列
        exer = pd.DataFrame.copy(raw_data).drop_duplicates(subset=['Problem Name'])
        exer['exer_id'] = exer['Problem Name'].apply(lambda x: exer2id[x])
        # 将字符串形式的assignment转化为数字形式的assignment_id
        unique_assignment = list(exer['assignment_name'].apply(lambda x: x[0]).unique())
        assignment2id, id2assignment = name_to_id(unique_assignment)
        exer['assignment_id'] = exer['assignment_name'].apply(lambda x: assignment2id[x[0]])
        # 将每个习题的KC(Default)->cpt_seq_name->cpt_seq，
        exer['cpt_seq_name'] = exer.apply(lambda x: preprocess_cpt_seq(x['KC(SubSkills)']), axis=1)
        cpt2id, id2cpt = name_to_id(kcs)
        exer['cpt_seq'] = exer['cpt_seq_name'].apply(lambda x: process_cpt_seq(x))
        df_exer = exer.reindex(columns=['exer_id', 'assignment_id', 'cpt_seq']).rename(
            columns={'exer_id': 'exer_id:token', 'assignment_id': 'assignment_id:token_seq',
                     'cpt_seq': 'cpt_seq:token_seq'})

        # 此处将数据保存到`cfg.MIDDATA_PATH`中
        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        # df_stu.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
        pd.set_option("mode.chained_assignment", "warn")  # ignore warning
        return
