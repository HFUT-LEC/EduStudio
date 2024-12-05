# M2C Atomic Data Operation List


|    M2C Atomic operation    | M2C Atomic Type | Description                                                  |
| :------------------------: | --------------- | ------------------------------------------------------------ |
|       M2C_Label2Int        | Data Cleaning   | Binarization for answering response                          |
|    M2C_FilterRecords4CD    | Data Cleaning   | Filter some students or exercises according specific conditions |
| M2C_FilteringRecordsByAttr | Data Cleaning   | Filtering Students without attribute values, Commonly used by Fair Models |
|        M2C_ReMapId         | Data Conversion | ReMap Column ID                                              |
|     M2C_BuildMissingQ      | Data Conversion | Build Missing Q-matrix                                       |
|   M2C_BuildSeqInterFeats   | Data Conversion | Build  sample format for Question-based KT                   |
|       M2C_CKCAsExer        | Data Conversion | Build  sample format for KC-based KT                         |
|   M2C_MergeDividedSplits   | Data Conversion | Merge train/valid/test set into one dataframe                |
|   M2C_RandomDataSplit4CD   | Data Partition  | Data partitioning for Cognitive Diagnosis                    |
|   M2C_RandomDataSplit4KT   | Data Partition  | Data partitioning for Knowledge Tracing                      |
|        M2C_GenKCSeq        | Data Generation | Generate Knowledge Component Sequence                        |
|        M2C_GenQMat         | Data Generation | Generate Q-matrix (i.e, exercise-KC relation)                |
|    M2C_BuildKCRelation     | Data Generation | Build Knowledge Component Relation Graph                     |
|     M2C_GenUnFoldKCSeq     | Data Generation | Generate Unfolded Knowledge Component Sequence               |
|      M2C_FillMissingQ      | Data Generation | Fill Missing Q-matrix                                        |

