# Dataset Status Protocol

In `Edustudio`, we view the dataset as three statuses: `rawdata`, `middata`, `cachedata`.
- inconsistent rawdata: the original data format provided by the dataset publisher.
- standardized middata: the standardized middle data format(see Middle Data Format Protocol) defined by EduStudio.
- model-friendly cachedata: the data format that is convenient for model usage.


## Dataset Folder Format Example

All datasets are required to store in a unified folder. The example below illustrated the `FrcSub` dataset folder format.

```
data/
├── FrcSub
│   ├── cachedata
│   │   ├── FrcSub_five_fold
│   │   │   ├── datatpl_cfg.json
│   │   │   ├── df_exer.pkl
│   │   │   ├── df_stu.pkl
│   │   │   ├── dict_test_folds.pkl
│   │   │   ├── dict_train_folds.pkl
│   │   │   ├── dict_valid_folds.pkl
│   │   │   └── final_kwargs.pkl
│   │   └── FrcSub_one_fold
│   │       ├── datatpl_cfg.json
│   │       ├── df_exer.pkl
│   │       ├── df_stu.pkl
│   │       ├── dict_test_folds.pkl
│   │       ├── dict_train_folds.pkl
│   │       ├── dict_valid_folds.pkl
│   │       └── final_kwargs.pkl
│   ├── middata
│   │   ├── FrcSub.exer.csv
│   │   └── FrcSub.inter.csv
│   └── rawdata
│       ├── data.txt
│       ├── problemdesc.txt
│       ├── qnames.txt
│       └── q.txt
```



## Dataset Stage protocol

The `rawdata` folder stores raw data files. For different datasets, they have different raw data file format.
There is an example that loading dataset from `rawdata`. 

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'load_data_from': "rawdata", # specify the loading stage of the dataset
        'raw2mid_op': 'R2M_FrcSub' # specify the R2M atomic operation 
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)
```

The `middata` folder stores middle data files. For existing datasets, we provide the atomic operation inheriting the protocol class `BaseRaw2Mid`, which process raw data to middle data. The middle is required in atomic file protocol.

There is an example that loading dataset from `middata`. 

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'load_data_from': "middata", # specify the loading stage of the dataset
        'is_save_cache': True, # whether to save cache data
        'cache_id': 'cache_default', # cache id, valid when is_save_cache=True
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)
```

With the `middata` of following atomic file protocol, we can implement some other atomic operations inheriting the protocol class `BaseMid2Cache` to build `cachedata` from `middata`. 


There is an example that loading dataset from `cachedata`. 

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'load_data_from': "cachedata", # specify the loading stage of the dataset
        'is_save_cache': False,
        'cache_id': 'cache_default', # cache id, valid when is_save_cache=True
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)
```


## Example：Atomic Operation Sequence of Data Processing

- R2M_FrcSub: process the Frcsub dataset from `rawdata` to `midata`
- M2C_FilterRecords4CD：Filter students or exercises whose number  of interaction records is less than a threshold
- M2C_ReMapId: ReMap feature Id
- M2C_RandomDataSplit4CD: Split Datasets
- M2C_GenQMat: Generate Q-matrix

The 'mid2cache_op_seq' option in datatpl_cfg specify the atomic operation sequence

```python
from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='FrcSub',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
    },
    datatpl_cfg_dict={
        'cls': 'CDInterExtendsQDataTPL',
        'load_data_from": "rawdata", # specify the loading stage of the dataset
        'raw2mid_op': 'R2M_FrcSub', 
        # the 'mid2cache_op_seq' option specify the atomic operation sequence
        'mid2cache_op_seq': ['M2C_Label2Int', 'M2C_FilterRecords4CD', 'M2C_ReMapId', 'M2C_RandomDataSplit4CD', 'M2C_GenQMat'],
    },
    modeltpl_cfg_dict={
        'cls': 'KaNCD',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL', 'InterpretabilityEvalTPL'],
    }
)
```
