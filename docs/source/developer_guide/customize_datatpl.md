# Customize Data Template

There are three common basic templates: `BaseDataTPL`, `GeneralDataTPL` and `EduDataTPL`. The `GeneralDataTPL` inherits from `BaseDataTPL` , supporting general functions such as `DataCache`、`Data Atomic Operation`、`Multiple Folds Training` and `DataFiles Protocol`, focusing on common interaction features. The `EduDataTPL` inherits from `GeneralDataTPL`, focusing on  features of  students and exercises on the basis of `GeneralDataTPL`. 

## BaseDataTPL

`BaseDataTPL` is affiliated with `Basic Architecture` and provide basic data protocols. All data templates should inherit the `BaseDataTPL`.

### Usage Scenario

If  a user is intended to abandon existing data processing method defined in `GeneralDataTPL`, the customized data template inheriting from `BaseDataTPL` is reasonable. The protocol in  `BaseDataTPL`  is  minimal, providing users with a wide range of playable space.

### Protocols

#### Description based protocols

- BaseDataTPL inherits from `torch.utils.data.Dataset`.

#### Variable based protocol

| name              | description                            |
| ----------------- | -------------------------------------- |
| default_cfg       | default configuration of data template |
| self.datatpl_cfg  | configuration of data template         |
| self.evaltpl_cfg  | configuration of evaluate template     |
| self.traintpl_cfg | configuration of training template     |
| self.modeltpl_cfg | configuration of model template        |
| self.frame_cfg    | configuration of framework             |
| self.logger       | logger object                          |

#### Function based protocol

| name            | description                                                  |
| --------------- | ------------------------------------------------------------ |
| from_cfg        | the entry point of create instance.                          |
| _check_params   | check rationality of configuration                           |
| get_extra_data  | return a dict object and the framework will pass this to model instance. |
| _copy           | copy method of current instance                              |
| get_default_cfg | return default_cfg  of current class and ancestral classes   |

#### Use Case

The best use case of `BaseDataTPL` is the `GenetalDataTPL`. Please see below.



## GeneralDataTPL

The `GeneralDataTPL` inherits from `BaseDataTPL`  and is affiliated with `Inherited Architecture`. It support general functions such as `DataCache`、`Data Atomic Operation`、`Multiple Folds Training` and `DataFiles Protocol`, focusing on common interaction features. 

### Usage Scenario

If new data template focuses on interaction features only and exploits existing functions (such as `DataCache`),  the customized data template inheriting from `BaseDataTPL` is appropriate.  

### Protocols

#### Description based protocols

- Data Cache, see the corresponding chapter
- Data Atomic Operation, see the corresponding chapter
- Data Files Protocol, see the corresponding chapter
- Multi Folds Training
  - The data template inherits from `torch.utils.data.Dataset` in pytorch. In the GeneralDataTPL, we set a status for current data template. 
  - Status of fold_id: specify current data template is served to which fold, the `self.dict_main` stores the current interaction data
  - Status of train/val/test/manager:  specify current data template is served to which stage. The manager status is the initial status, and other status is a copied object of manager status.

#### Variable based protocol

| name                  | description                                                  |
| --------------------- | ------------------------------------------------------------ |
| self.common_str2df    | The dictionary object read from files will be passed into the sequence of atomic operations |
| self.hasValidDataset  | Under the train/val/test setting, determine if a validation set exists |
| self.df               | If dataset is not divided, self.df will store the dataframe object from interaction csv file |
| self.df_train         | If dataset is divided, self.df will store the dataframe object from training interaction csv file |
| self.df_valid         | If dataset is divided, self.df will store the dataframe object from validation interaction csv file |
| self.df_test          | If dataset is divided, self.df will store the dataframe object from test interaction csv file |
| self.status           | Store current status of current template including fold_id and train/val/test status |
| self.df_train_folds   | Store the dataframe object of training data of each fold     |
| self.df_valid_folds   | Store the dataframe object of validation data of each fold   |
| self.df_test_folds    | Store the dataframe object of test data of each fold         |
| self.dict_train_folds | Store the dictionary object of training data of each fold    |
| self.dict_valid_folds | Store the dictionary object of validation data of each fold  |
| self.dict_test_folds  | Store the dictionary object of test data of each fold        |
| self.dict_main        | store the dictionary object of  current status (i.e., train/val/test) |

#### Function based protocol

| name              | description                                                  |
| ----------------- | ------------------------------------------------------------ |
| load_data         | load data files into python object                           |
| process_data      | process middata                                              |
| build_datasets    | copy current data template into multiple dataset objects     |
| build_dataloaders | build data loaders from multiple dataset objects             |
| save_cache        | save cache process                                           |
| check_cache       | check if the imported cache matches the current settings     |
| load_cache        | load cache process                                           |
| collate_fn        | collate function when build data loaders                     |
| \_\_len_\_        | defined in pytorch, get the length of samples                |
| \_\_getitem_\_    | defined in pytorch, get the specific sample                  |
| df2dict           | convert self.df_train/val/test_folds into self.dict_train/val/test_folds |
| set_info_for_fold | set current data object when a fold_id is specified          |

### Use Cases

The best use case of `GeneralDataTPL` is the `EduDataTPL`. Please see below.

## EduDataTPL

 The `EduDataTPL` inherits from `GeneralDataTPL`, focusing on  features of  students and exercises on the basis of `GeneralDataTPL`. 

### Usage Scenarios

On the basis of GeneralDataTPL, the `EduDataTPL`  additionally considers the features of students and exercises. Following Atomic Files Protocol, `EduDataTPL`  will read `.stu.csv` and `.exer.csv` file to load the data. 

### Protocols

#### Description based protocols

- The features of students and exercises would be treated as the extra data, which means that the extra data would be pass to model as mentioned in `BaseDataTPL` protocol.

#### Variable based protocol

| name              | description                                     |
| ----------------- | ----------------------------------------------- |
| self.df_stu       | Store the dataframe object of student features  |
| self.df_exer      | Store the dataframe object of exercise features |
| self.hasStuFeats  | Determine if student features  exists           |
| self.hasExerFeats | Determine if exercise features  exists          |
| self.hasQmat      | Determine if Q-matrix  exists                   |

### Use Cases

The use cases of `EduDataTPL` please see other data templates inheriting from `EduDataTPL` .

