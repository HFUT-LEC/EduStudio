# Inheritable Default Configuration

The management of default configuration in Edustudio is implemented by class variable, i.e. a dictionary object called default_config. 

Templates usually introduce new features through inheritance, and these new features may require corresponding configurations, so the default configuration we provide is inheritable.

## Example

The inheritance example of data template is illustrated as follows. We present an example in the data preparation procedure. There are three data template classes (DataTPLs) that inherit from each other: BaseDataTPL, GeneralDataTPL, and EduDataTPL. If users specify current DataTPL is EduDataTPL, the eventual default\_config of data preparation procedure is a merger of default\_cfg of three templates. When a configuration conflict is encountered, the default\_config of subclass template takes precedence over that of parent class templates. As a result, other configuration portals (i.e, configuration file, parameter dictionary, and command line) can only specify parameters that are confined within the default configuration. The advantage of the inheritable design is that it facilitates the reader to locate the numerous hyperparameters.

```python
class BaseDataTPL(Dataset):
    default_cfg = {'seed': 2023}


class GeneralDataTPL(BaseDataTPL):
    default_cfg = {
        'seperator': ',',
        'n_folds': 1,
        'is_dataset_divided': False,
        'is_save_cache': False,
        'cache_id': 'cache_default',
        'load_data_from': 'middata', # ['rawdata', 'middata', 'cachedata']
        'inter_exclude_feat_names': (),
        'raw2mid_op': None, 
        'mid2cache_op_seq': []
    }


class EduDataTPL(GeneralDataTPL):
    default_cfg = {
        'exer_exclude_feat_names': (),
        'stu_exclude_feat_names': (),
    }
```

If the currently specified data template is `EduDataTPL`,  then the framework will get the final `default_cfg` through API `get_default_cfg`, which would be:

```python
default_cfg = {
    'exer_exclude_feat_names': (),
    'stu_exclude_feat_names': (),
    'seperator': ',',
    'n_folds': 1,
    'is_dataset_divided': False,
    'is_save_cache': False,
    'cache_id': 'cache_default',
    'load_data_from': 'middata', # ['rawdata', 'middata', 'cachedata']
    'inter_exclude_feat_names': (),
    'raw2mid_op': None, 
    'mid2cache_op_seq': [],
    'seed': 2023
}
```

The final `default_cfg` follows two rules:

- The subclass would incorporate the `default_cfg` of all parent classes.
- When a conflict happened for the same key, the subclass would dominate the priority.
