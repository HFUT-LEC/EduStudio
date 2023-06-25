# Dataset Stage Protocol

In `Edustudio`, we view the dataset as three stages: `rawdata`, `middata`, `cachedata`.


## Dataset Folder Format Example

All datasets are required to store in a unified folder. The below example illustrated the `FrcSub` dataset folder format.

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

The `middata` folder stores middle data files. For existing datasets, we provide the atomic operation inheriting the protocol class `BaseRaw2Mid`, which process raw data to middle data. The middle is required in atomic file protocol.

With the `middata` of following atomic file protocol, we can implement some other atomic operations inheriting the protocol class `BaseMid2Cache` to build `cachedata` from `middata`. 



## Example：Atomic Operation Sequence of Data Processing

- R2M_FrcSub: process the Frcsub dataset from `rawdata` to `midata`

- M2C_FilterRecords4CD：Filter students or exercises whose number  of interaction records is less than a threshold
- M2C_ReMapId: ReMap feature Id
- M2C_RandomDataSplit4CD: Split Datasets
- M2C_GenQMat: Generate Q-matrix





