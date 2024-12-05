from .base_evaltpl import BaseEvalTPL
import pandas as pd
import numpy as np
from edustudio.utils.common import tensor2npy


class FairnessEvalTPL(BaseEvalTPL):
    """Fairness Cognitive Evaluation
    """
    default_cfg = {
        'use_sensi_attrs': ['gender:token'],
        'use_metrics': ['EO', 'DP', 'FCD']
    }

    def _check_params(self):
        assert len(set(self.evaltpl_cfg[self.__class__.__name__]['use_metrics']) - {'EO', 'DP', 'FCD'}) == 0

    def eval(self, **kwargs):
        stu_id = tensor2npy(kwargs['stu_id'])
        pd_soft = tensor2npy(kwargs['y_pd'])
        gt = tensor2npy(kwargs['y_gt'])
        pd_hard = (pd_soft >= 0.5).astype(np.int64)

        df_stu = self.extra_data['df_stu']

        df = pd.DataFrame()
        df['stu_id:token'] = stu_id
        df['pd_soft'] = pd_soft
        df['pd_hard'] = pd_hard
        df['gt'] = gt
        df = df.merge(df_stu, on='stu_id:token', how='left')
        
        ret_dic = {}
        for attr in self.evaltpl_cfg[self.__class__.__name__]['use_sensi_attrs']:
            g_names = df_stu[attr].unique()

            for use_metric in self.evaltpl_cfg[self.__class__.__name__]['use_metrics']:
                if len(g_names) == 2:
                    if use_metric == 'EO': ret_dic[f"EO_{attr}"] = self.get_eo(df, attr)
                    if use_metric == 'DP': ret_dic[f"DP_{attr}"] = self.get_dp(df, attr)
                    if use_metric == 'FCD': ret_dic[f"FCD_{attr}"] = self.get_fcd(df, attr)
                else:
                    pass
        return ret_dic

    def get_dp(self, df, sensitive_attr):
        """Demographic Parity
        """
        dp = df.groupby(sensitive_attr)['pd_hard'].mean()
        return abs(dp[0] - dp[1])

    def get_eo(self, df, sensitive_attr):
        """Equal Opportunity
        """
        eo = df.groupby([sensitive_attr, 'gt'])['pd_hard'].mean()
        return abs(eo[0][1] - eo[1][1])


    def get_fcd(self, df, sensitive_attr):
        """Fair Cognitive Diagnosis [1]
        [1]zhang zheng, et al, Understanding and Improving Fairness in Cognitive Diagnosis,  SCIENCE CHINA Information Sciences, 2023, ISSN 1674-733X, https://doi.org/10.1007/s11432-022-3852-0.
        """
        fcd_pd = df.groupby([sensitive_attr, 'stu_id:token'])['pd_hard'].mean()
        fcd_pd = fcd_pd[0].mean() - fcd_pd[1].mean()

        fcd_gt = df.groupby([sensitive_attr, 'stu_id:token'])['gt'].mean()
        fcd_gt = fcd_gt[0].mean() - fcd_gt[1].mean()
        return abs(fcd_pd - fcd_gt)


    def add_extra_data(self, **kwargs):
        self.extra_data = kwargs

        df_stu = self.extra_data['df_stu']
        assert df_stu is not None
        for attr in self.evaltpl_cfg[self.__class__.__name__]['use_sensi_attrs']:
            assert attr in df_stu
            g_names = df_stu[attr].unique()

            for use_metric in self.evaltpl_cfg[self.__class__.__name__]['use_metrics']:
                assert len(g_names) >= 2
                if len(g_names) > 2:
                    if use_metric in {'EO', 'DP', 'FCD'}:
                        self.logger.warning(f"As the number of sensitive attribute `{attr}` values > 2, the fairness metric {use_metric} is not supported for the `{attr}`")
