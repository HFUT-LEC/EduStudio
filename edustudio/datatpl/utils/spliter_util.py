import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


class SpliterUtil:
    @staticmethod
    def divide_data_df_one_fold(df, divide_scale_list, seed, label_field=None, shuffle=True):
        n_split_1, n_split_2 = None, None
        if len(divide_scale_list) == 2:
            n_split_1 = np.sum(divide_scale_list) / divide_scale_list[-1]
        elif len(divide_scale_list) == 3:
            n_split_1 = np.sum(divide_scale_list) / divide_scale_list[-1]
            n_split_2 = np.sum(divide_scale_list[0:2]) / divide_scale_list[1]
        else:
            raise ValueError(f"unsupported divide_scale_list={divide_scale_list}")

        if label_field:
            skf = StratifiedKFold(n_splits=int(n_split_1),
                                shuffle=shuffle, random_state=seed)
            splits = skf.split(df, df[label_field])
        else:
            skf = KFold(n_splits=int(n_split_1),
                                shuffle=shuffle, random_state=seed)
            splits = skf.split(df)

        train_df, val_df, test_df = None, None, None
        for train_index, test_index in splits:
            train_df_tmp = df.iloc[train_index].reset_index(drop=True)
            test_df = df.iloc[test_index].reset_index(drop=True)
            break

        if n_split_2 is not None:
            if label_field:
                skf = StratifiedKFold(n_splits=int(n_split_2),
                                    shuffle=True, random_state=seed)
                splits = skf.split(train_df_tmp, train_df_tmp[label_field])
            else:
                skf = KFold(n_splits=int(n_split_2),
                                    shuffle=True, random_state=seed)
                splits = skf.split(train_df_tmp)
            for train_index, val_index in splits:
                train_df = train_df_tmp.iloc[train_index].reset_index(
                    drop=True)
                val_df = train_df_tmp.iloc[val_index].reset_index(drop=True)
                break
        else:
            train_df = train_df_tmp
        return train_df, val_df, test_df
    
    @staticmethod
    def divide_data_df_multi_folds(df, n_folds, seed, label_field=None, shuffle=True):
        if label_field:
            skf = StratifiedKFold(n_splits=int(n_folds),
                                shuffle=shuffle, random_state=seed)
            splits = skf.split(df, df[label_field])
        else:
            skf = KFold(n_splits=int(n_folds),
                                shuffle=shuffle, random_state=seed)
            splits = skf.split(df)

        train_df_list, test_df_list = [], []
        for train_index, test_index in splits:
            train_df = df.iloc[train_index].reset_index(drop=True)
            test_df = df.iloc[test_index].reset_index(drop=True)
            train_df_list.append(train_df)
            test_df_list.append(test_df)

        return train_df_list, test_df_list

    @staticmethod
    def divide_data_dict(dic, divide_scale_list, seed, label_field=None, shuffle=True):
        keys = list(dic.keys())
        n_split_1, n_split_2 = None, None
        if len(divide_scale_list) == 2:
            n_split_1 = np.sum(divide_scale_list) / divide_scale_list[-1]
        elif len(divide_scale_list) == 3:
            n_split_1 = np.sum(divide_scale_list) / divide_scale_list[-1]
            n_split_2 = np.sum(divide_scale_list[0:2]) / divide_scale_list[1]
        else:
            raise ValueError(f"unsupported divide_scale_list={divide_scale_list}")

        if label_field:
            skf = StratifiedKFold(n_splits=int(n_split_1),
                                shuffle=shuffle, random_state=seed)
            splits = skf.split(dic[keys[0]], dic[label_field])
        else:
            skf = KFold(n_splits=int(n_split_1),
                                shuffle=shuffle, random_state=seed)
            splits = skf.split(dic[keys[0]])

        train_dic, val_dic, test_dic = None, None, None
        for train_index, test_index in splits:
            train_dic_tmp = {dic[k][train_index] for k in keys}
            test_dic = {dic[k][test_index] for k in keys}
            break

        if n_split_2 is not None:
            if label_field:
                skf = StratifiedKFold(n_splits=int(n_split_2),
                                    shuffle=True, random_state=seed)
                splits = skf.split(train_dic_tmp[keys[0]], train_dic_tmp[label_field])
            else:
                skf = KFold(n_splits=int(n_split_2), shuffle=True, random_state=seed)
                splits = skf.split(train_dic_tmp)
            for train_index, val_index in splits:
                train_dic = {train_dic_tmp[k][train_index] for k in keys}
                val_dic = {train_dic_tmp[k][val_index] for k in keys}
                break
        else:
            train_dic = train_dic_tmp
        return train_dic, val_dic, test_dic
