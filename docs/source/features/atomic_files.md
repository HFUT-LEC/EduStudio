# Middle Data Format Protocol

In `EduStudio`, we adopt a flexible CSV (Comma-Separated Values) file format following [Recbole](https://recbole.io/atomic_files.html).  The flexible CSV format is defined in `middata` stage of dataset (see dataset stage protocol for details).

The Middle Data Format Protocol including two parts:  `Columns name Format` and `Filename Format`.

## Columns Name Format

| feat_type     | Explanations                | Examples                          |
| ------------- | --------------------------- | --------------------------------- |
| **token**     | single discrete feature     | exer_id, stu_id                   |
| **token_seq** | discrete features sequence  | knowledge concept seq of exercise |
| **float**     | single continuous feature   | label, start_timestamp            |
| **float_seq** | continuous feature sequence | word2vec embedding of exercise    |



## Filename format

So far, there are five atomic files in edustudio.

**Note**: Users could also load other types of data except the three atomic files below. `{dt}` is the dataset name.

| filename format      | description                                          |
| -------------------- | ---------------------------------------------------- |
| {dt}.inter.csv       | Student-Exercise Interaction data                    |
| {dt}.train.inter.csv | Student-Exercise Interaction data for training set   |
| {dt}.train.inter.csv | Student-Exercise Interaction data for validation set |
| {dt}.train.inter.csv | Student-Exercise Interaction data for test set       |
| {dt}.stu.csv         | Features of students                                 |
| {dt}.exer.csv        | Features of exercises                                |



## Example

###  example_dt.inter.csv

| stu_id:token | exer_id:token | label:float |
| ------------ | ------------- | ----------- |
| 0            | 1             | 0.0         |
| 1            | 0             | 1.0         |

### example_dt.stu.csv

| stu_id:token | gender:token | occupation:token |
| ------------ | ------------ | ---------------- |
| 0            | 1            | 11               |
| 1            | 0            | 7                |

### example_dt.exer.csv

| exer_id:token | cpt_seq:token_seq | w2v_emb:float_seq      |
| ------------- | ----------------- | ---------------------- |
| 0             | [0, 1]            | [0.121, 0.123, 0.761]  |
| 1             | [1, 2, 3]         | [0.229, -0.113, 0.138] |
