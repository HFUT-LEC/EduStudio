# Atomic Operations

In `Edustudio`, we view the dataset from three stages: `rawdata`, `middata`, `cachedata`.

 we treat the whole data processing as multiple atomic operations called atomic operation sequence. 
The first atomic operation, inheriting the protocol class `BaseRaw2Mid`, is the process from raw data to middle data.
The following atomic operations, inheriting the protocol class `BaseMid2Cache`,  construct the process from middle data to cache data.

The atomic operation protocol can be seen at `Atomic Operation Protocol`.



## Atomic Operation Table

In the following, we give a table to display existing atomic operations.

### Raw2Mid

| name            | description                                                  |
| --------------- | ------------------------------------------------------------ |
| R2M_ASSIST_0910 | The atomic operation that process the Assistment_0910 dataset from rawdata into midata |
| R2M_FrcSub      | The atomic operation that process the FrcSub dataset from rawdata into midata |
| R2M_ASSIST_1213 | The atomic operation that process the Assistment_1213 dataset from rawdata into midata |
| R2M_Math1       | The atomic operation that process the Math1dataset from rawdata into midata |
| R2M_Math2       | The atomic operation that process the Math2 dataset from rawdata into midata |
| R2M_AAAI_2023   | The atomic operation that process the AAAI 2023 challenge dataset from rawdata into midata |
| R2M_Algebra_0506 | The atomic operation that process the Algebra 2005-2006 dataset from rawdata into midata |
| R2M_ASSIST_1516 | The atomic operation that process the Assistment 2015-2016 dataset from rawdata into midata |

### Mid2Cache

#### common

| name                   | description                                   |
| ---------------------- | --------------------------------------------- |
| M2C_Label2Int          | convert label column into discrete values     |
| M2C_MergeDividedSplits | merge train/valid/test set into one dataframe |
| M2C_ReMapId            | ReMap Column ID                               |
| M2C_GenQMat            | Generate Q-matrix                             |

#### CD

| name                   | description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| M2C_RandomDataSplit4CD | Split datasets Randomly for CD                               |
| M2C_FilterRecords4CD   | Filter students or exercises whose number of interaction records is less than a threshold |

#### KT

| name                   | description                                 |
| ---------------------- | ------------------------------------------- |
| M2C_BuildSeqInterFeats | Build Sequential Features and Split dataset |
| M2C_KCAsExer          | Treat knowledge concept as exercise         |
| M2C_GenKCSeq          | Generate knowledge concept seq              |
| M2C_GenUnFoldKCSeq    | Unfold knowledge concepts                   |

