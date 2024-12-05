from edustudio.quickstart import run_edustudio

class TestRun:
    def test_cd(self):
        print("---------- test cd ------------")
        run_edustudio(
            dataset='FrcSub',
            cfg_file_name=None,
            traintpl_cfg_dict={
                'cls': 'GeneralTrainTPL',
                'epoch_num': 2,
                'device': 'cpu'
            },
            datatpl_cfg_dict={
                'cls': 'CDInterExtendsQDataTPL'
            },
            modeltpl_cfg_dict={
                'cls': 'KaNCD',
            },
            evaltpl_cfg_dict={
                'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
            }
        )
