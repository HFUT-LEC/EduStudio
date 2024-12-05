# Customize Atomic Operation

In EduStudio, we treat the whole data processing as multiple atomic operations called atomic operation sequence. 
The first atomic operation, inheriting the protocol class `BaseRaw2Mid`, is the process from raw data to middle data.
The following atomic operations, inheriting the protocol class `BaseMid2Cache`,  construct the process from middle data to cache data.

## BaseRaw2Mid

The atomic operations inheriting `BaseRaw2Mid` preprocess the raw dataset into middle dataset (standardized data files).

### Protocols

The protocols in `BaseRaw2Mid` are listed as follows:

| name         | description                                    | type               | note                    |
| ------------ | ---------------------------------------------- | ------------------ | ----------------------- |
| self.dt      | current dataset name                           | instance variable  | given in BaseRaw2Mid    |
| self.rawpath | raw data path of current dataset               | instance variable  | given in BaseRaw2Mid    |
| self.midpath | middle data path of current dataset            | instance variable  | given in BaseRaw2Mid    |
| self.logger  | logger object                                  | instance variable  | given in BaseRaw2Mid    |
| process      | preprocess the raw dataset into middle dataset | function interface | implemented by subclass |

### Example

The following example illustrates the process of `Assistment 2019-2010 ` dataset from raw data to middle data.

```python
class R2M_ASSIST_0910(BaseRaw2Mid):
    def process(self):
        df = pd.read_csv(f"{self.rawpath}/skill_builder_data.csv", encoding='ISO-8859-1')

        ......

        df_inter.to_csv(f"{self.midpath}/{self.dt}.inter.csv", index=False, encoding='utf-8')
        df_user.to_csv(f"{self.midpath}/{self.dt}.stu.csv", index=False, encoding='utf-8')
        df_exer.to_csv(f"{self.midpath}/{self.dt}.exer.csv", index=False, encoding='utf-8')
```

The above function first read raw data file from specific folder path (i.e., self.rawpath). After processing middle data, it will save middle data in specific folder path (i.e., self.midpath).

## BaseMid2Cache

The atomic operations inheriting `BaseMid2Cache` preprocess the middle dataset into cache dataset (standardized data files). Different from  atomic operations inheriting `BaseRaw2Mid`, in one atomic operation sequence, atomic operations inheriting  `BaseRaw2Mid` should be unique and be the first position. Atomic operations inheriting `BaseMid2Cache` could be multiple and dominate following all operations.

### Protocols

The protocols in `BaseMid2Cache` are listed as follows:

| name          | description                                                       | type               | note                    |
| ------------- | ----------------------------------------------------------------- | ------------------ | ----------------------- |
| default_cfg   | the default configuration of operation                            | class variable     |                         |
| self.logger   | logger object                                                     | instance variable  | given in BaseMid2Cache  |
| self.m2c_cfg  | actual configuration in running process                           | instance variable  | given in BaseMid2Cache  |
| _check_params | check rationality of configuration                                | function interface | implemented by subclass |
| process       | preprocess the raw dataset into middle dataset                    | function interface | implemented by subclass |
| set_dt_info   | store dataset information in the process (such as student number) | function interface | implemented by subclass |

### Example

The following example illustrates the partial process code of `M2C_RandomDataSplit4CD` atomic operation, which splits datasets for cognitive diagnosis.

```python
class M2C_RandomDataSplit4CD(BaseMid2Cache):
    default_cfg = {
        'seed': 2023,
        "divide_scale_list": [7,1,2],
    }

    def _check_params(self):
        super()._check_params()
        assert 2 <= len(self.m2c_cfg['divide_scale_list']) <= 3
        assert sum(self.m2c_cfg['divide_scale_list']) == 10

    def process(self, **kwargs):
        df = kwargs['df']

        if self.n_folds == 1:
            assert kwargs.get("df_train", None) is None
            assert kwargs.get("df_valid", None) is None
            assert kwargs.get("df_test", None) is None
            df_train, df_valid, df_test = self.one_fold_split(df)
            kwargs['df_train_folds'] = [df_train]
            kwargs['df_valid_folds'] = [df_valid] if df_valid is not None else []
            kwargs['df_test_folds'] = [df_test]
        else:
            df_train_list, df_test_list = self.multi_fold_split(df)
            kwargs['df_train_folds'] = df_train_list
            kwargs['df_test_folds'] = df_test_list

        return kwargs

    def set_dt_info(self, dt_info, **kwargs):
        if 'stu_id:token' in kwargs['df'].columns:
            dt_info['stu_count'] = int(kwargs['df']['stu_id:token'].max() + 1)
```
