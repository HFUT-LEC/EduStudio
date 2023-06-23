from .raw2mid import BaseRaw2Mid
import pandas as pd
import json


class R2M_AAAI_2023(BaseRaw2Mid):
    def get_df(self, df_f):
        records = []
        exerid_dic = {}  # 用来检查重复的习题
        ex_uid = -1
        for index, row in df_f.iterrows():  # 遍历csv所有行
            #         row_indexs.append(index)
            user_id = row["uid"]
            exer_id_s = row["questions"]
            score_s = row["responses"]
            time_s = row["timestamps"]
            #     print(exer_id_s)
            #     print(score_s)
            e_s = exer_id_s.split(",")
            s_s = score_s.split(",")
            t_s = time_s.split(",")
            if ex_uid != user_id:  # 另外一个学生了
                exerid_dic = {}
            for index, e in enumerate(e_s):
                e = int(e)
                if e == -1: break
                re_dict = {}  # 用来保存一条交互记录
                if not e in exerid_dic.keys():
                    exerid_dic[e] = 1
                    re_dict["user_id"] = user_id
                    re_dict["exer_id"] = e
                    re_dict["score"] = int(s_s[index])
                    re_dict["time"] = t_s[index]
                    # for i in n_q:
                    #     if i["exer_id"] == e:
                    #         re_dict["knowledge_code"]= i["knowledge_code"]
                    #         re_dict['content'] = i["text"]
                    #         break
                    # df.loc[len(df)] = [re_dict["user_id"],re_dict["exer_id"],re_dict["score"],re_dict["time"]]
                    records.append([re_dict["user_id"], re_dict["exer_id"], re_dict["score"], re_dict["time"]])
        df = pd.DataFrame(records, columns=['user_id', 'exer_id', 'label', 'start_timestamp'])
        return df

    def get_kw_text(self, q, key2id):
        n_q = []
        for k, v in q.items():  #
            t = {}
            t["exer_id"] = key2id["questions"][k]

            #     "concept_routes": ["2273----1425----637----297", "121----1425----637----297"],
            concept_routes = v["concept_routes"]
            ls_kw = []
            for kw in concept_routes:
                s = kw.split("----")[-1]
                ls_kw.append(key2id["concepts"][s])
            t["knowledge_code"] = list(set(ls_kw))
            t["text"] = v["content"]
            n_q.append(t)
        return n_q

    def process(self):
        df_ = pd.read_csv(f"{self.rawpath}/train_valid_sequences.csv", low_memory=False)
        df_f = df_[df_["fold"] == 0]
        with open(f"{self.rawpath}/questions.json", encoding='utf8') as i_f:
            q = json.load(i_f)
        with open(f"{self.rawpath}/keyid2idx.json", encoding='utf8') as i_f:
            key2id = json.load(i_f)

        n_q = self.get_kw_text(q, key2id)
        df = self.get_df(df_f)
        grouped = df.groupby('user_id', as_index=False)

        # 定义一个函数来修改每个group的DataFrame
        def modify_group(group):
            group.sort_values(['start_timestamp'], inplace=True)
            group['order_id'] = range(len(group))
            return group

        # 使用apply函数来应用修改函数到每个group的DataFrame
        df_modified = grouped.apply(modify_group)
        df = df_modified.reset_index(drop=True)
        df_inter = df[['exer_id', 'user_id', 'start_timestamp', 'order_id', 'label']]
        df_inter = df_inter.rename(
            columns={'user_id': 'stu_id:token', 'exer_id': 'exer_id:token', 'label': 'label:float',
                     'start_timestamp': 'start_timestamp:float',
                     'order_id': 'order_id:token'})
        df_exer = df[['exer_id']]
        df_exer = df_exer.drop_duplicates(subset=['exer_id'])
        records = []
        for e in df_exer['exer_id'].to_list():
            re_dict = {}
            for i in n_q:
                if i["exer_id"] == e:
                    re_dict["knowledge_code"] = i["knowledge_code"]
                    re_dict['content'] = i["text"]
                    break
            records.append([e, re_dict["knowledge_code"], re_dict['content']])
        df_exer = pd.DataFrame(records, columns=['exer_id:token', 'cpt_seq:token_seq', 'content:token_seq'])

        # cpt_ls = []
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].apply(lambda x: ",".join([str(i) for i in x]))
        df_exer['content:token_seq'] = df_exer['content:token_seq'].apply(lambda x: ",".join([str(i) for i in x]))
        # ser_cpt = df['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        # ser_cpt.to_list()
        # cpt_set = set(cpt_ls)
        # # 此处统计数据集，保存到cfg对象中
        # cfg.exer_count = len(df['exer_id'].unique())
        # cfg.stu_count = len(df['user_id'].unique())
        # cfg.cpt_count = len(cpt_set)
        # cfg.interaction_count = len(df_inter)

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
