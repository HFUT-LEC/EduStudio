# Atomic Data Operation Protocol

In `Edustudio`, we view the dataset from three stages: `rawdata`, `middata`, `cachedata`.

We treat the whole data processing as multiple atomic operations called atomic operation sequence. 
The first atomic operation, inheriting the protocol class `BaseRaw2Mid`, is the process from raw data to middle data.
The following atomic operations, inheriting the protocol class `BaseMid2Cache`,  construct the process from middle data to cache data.


## Partial Atomic Operation Table

In the following, we give a table to display some existing atomic operations. For more detailed Atomic Operation Table, please see the `user_guide/Atomic Data Operation List`

### Raw2Mid

For the conversion from rawdata to middata, we implement a specific atomic data operation prefixed with `R2M` for each dataset.

| name            | Corresponding datase                                                |
| --------------- | ------------------------------------------------------------ |
| R2M_ASSIST_0910 |  ASSISTment 2009-2010  |
| R2M_FrcSub      | Frcsub |
| R2M_ASSIST_1213 | ASSISTment 2012-2013  |
| R2M_Math1       | Math1 |
| R2M_Math2       | Math2 |
| R2M_AAAI_2023   | AAAI 2023 Global Knowledge Tracing Challenge |
| R2M_Algebra_0506 | Algebra 2005-2006 |
| R2M_ASSIST_1516 | ASSISTment 2015-2016 |

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

