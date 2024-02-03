from ..common.base_mid2cache import BaseMid2Cache
from sklearn.preprocessing import LabelEncoder
from itertools import chain


class M2C_KCAsExer(BaseMid2Cache):
    """Knowledge Concept As Exercise
    """
    default_cfg = {}

    def process(self, **kwargs):
        df = kwargs['df']
        df_exer = kwargs['df_exer']
        df_train, df_valid, df_test = kwargs['df_train'], kwargs['df_valid'], kwargs['df_test']

        if df is not None:
            assert df_train is None and df_valid is None and df_test is None
            kwargs['df_exer'], kwargs['df'] = self.cpt_as_exer_4undivided(df_exer, df)

        else:
            assert df_train is not None and df_test is not None
            kwargs['df_exer'], kwargs['df_train'], kwargs['df_valid'], kwargs['df_test'] = self.cpt_as_exer_4divided(df_exer, df_train, df_valid, df_test)
        
        return kwargs

    @staticmethod
    def cpt_as_exer_4undivided(df_exer, df):
        df_exer.rename(columns={"exer_id:token": "exer_id_old:token"}, inplace=True)
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].apply(lambda x: "_".join(list(map(lambda i: str(i), set(x)))))
        lbe = LabelEncoder()
        df_exer['cpt_seq:token_seq'] = lbe.fit_transform(df_exer['cpt_seq:token_seq']).tolist()
        df_exer.rename(columns={"cpt_seq:token_seq": "exer_id:token"}, inplace=True)

        df.rename(columns={'exer_id:token': 'exer_id_old:token'}, inplace=True)
        df = df.merge(df_exer[['exer_id_old:token', "exer_id:token"]], on='exer_id_old:token', how='left').reset_index(drop=True)
        df.drop(labels=['exer_id_old:token'], axis=1, inplace=True)
        return df_exer, df

    @staticmethod
    def cpt_as_exer_4divided(df_exer, df_train, df_valid, df_test):
        df_exer.rename(columns={"exer_id:token": "exer_id_old:token"}, inplace=True)
        df_exer['cpt_seq:token_seq'] = df_exer['cpt_seq:token_seq'].apply(
            lambda x: "_".join(list(map(lambda i: str(i), set(x))))
        )
        lbe = LabelEncoder()
        df_exer['cpt_seq:token_seq'] = lbe.fit_transform(df_exer['cpt_seq:token_seq']).tolist()
        df_exer.rename(columns={"cpt_seq:token_seq": "exer_id:token"}, inplace=True)

        df_train.rename(columns={'exer_id:token': 'exer_id_old:token'}, inplace=True)
        df_train = df_train.merge(df_exer, on='exer_id_old:token', how='left').reset_index(drop=True)
        df_train.drop(labels=['exer_id_old:token'], axis=1, inplace=True)
        if df_valid is not None:
            df_valid.rename(columns={'exer_id:token': 'exer_id_old:token'}, inplace=True)
            df_valid = df_valid.merge(df_exer, on='exer_id_old:token', how='left').reset_index(drop=True)
            df_valid.drop(labels=['exer_id_old:token'], axis=1, inplace=True)
        df_test.rename(columns={'exer_id:token': 'exer_id_old:token'}, inplace=True)
        df_test = df_test.merge(df_exer[['exer_id_old:token', "exer_id:token"]], on='exer_id_old:token', how='left').reset_index(drop=True)
        df_test.drop(labels=['exer_id_old:token'], axis=1, inplace=True)

        return df_exer, df_train, df_valid, df_test

    def set_dt_info(self, dt_info, **kwargs):
        if 'stu_id:token' in kwargs['df'].columns:
            dt_info['stu_count'] = int(kwargs['df']['stu_id:token'].max() + 1)
        if 'exer_id:token' in kwargs['df'].columns:
            dt_info['exer_count'] = int(kwargs['df']['exer_id:token'].max() + 1)
        if kwargs.get('df_exer', None) is not None:
            if 'cpt_seq:token_seq' in kwargs['df_exer']:
                dt_info['cpt_count'] = len(set(list(chain(*kwargs['df_exer']['cpt_seq:token_seq'].to_list()))))

    