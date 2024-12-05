from edustudio.utils.common import UnifyConfig
import os
from enum import Enum
import pandas as pd
from .base_datatpl import BaseDataTPL
import copy
from torch.utils.data import Dataset, DataLoader, default_collate
import pickle
import json
from deepdiff import DeepDiff
import importlib
import torch


class DataTPLMode(Enum):
    TRAIN=1
    VALID=2
    TEST=3
    MANAGER=4


class DataTPLStatus(object):
    def __init__(self, mode=DataTPLMode.MANAGER, fold_id=None) -> None:
        self.mode = mode
        self.fold_id = fold_id
        

class GeneralDataTPL(BaseDataTPL):
    """General Data Template
    """
    default_cfg = {
        'seperator': ',',
        'n_folds': 1,
        'is_dataset_divided': False,
        'is_save_cache': False,
        'cache_id': 'cache_default',
        'load_data_from': 'middata', # ['rawdata', 'middata', 'cachedata']
        'inter_exclude_feat_names': (),
        'raw2mid_op': "None", 
        'mid2cache_op_seq': []
    }

    def __init__(
                self, cfg:UnifyConfig,
                df: pd.DataFrame=None,
                df_train: pd.DataFrame=None,
                df_valid: pd.DataFrame=None,
                df_test: pd.DataFrame=None,
                status: DataTPLStatus=DataTPLStatus()
            ):
        super().__init__(cfg)
        self.df = df
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.status = status
        self.dict_main = None

        self.df_train_folds = []
        self.df_valid_folds = []
        self.df_test_folds = []
        self.dict_train_folds = []
        self.dict_valid_folds = []
        self.dict_test_folds = []

        # self.r2m_op = self._get_r2m_op()
        self.m2c_op_list = self._get_m2c_op_list()

        self.datatpl_cfg['dt_info'] = {}
        if self.datatpl_cfg['load_data_from'] == 'cachedata':
            self.load_cache()
            self.check_cache()
            self.process_data()
            self.logger.info(f"Load from cache successfully: {self.datatpl_cfg['cache_id']}")
            self.logger.info(self.datatpl_cfg['dt_info'])
        else:
            # self.feat2type = {}
            self.process_data()
            self.df2dict()
            if self.datatpl_cfg['is_save_cache']:
                self.save_cache()
            self.logger.info(self.datatpl_cfg['dt_info'])

    @classmethod
    def from_cfg(cls, cfg):
        """an interface to instantiate a data template

        Args:
            cfg (UnifyConfig): the global config object

        Returns:
           BaseDataTPL
        """
        if not os.path.exists(cfg.frame_cfg.data_folder_path) or len(os.listdir(cfg.frame_cfg.data_folder_path)) == 0:
            cls.download_dataset(cfg)
        
        load_data_from = cfg.datatpl_cfg['load_data_from']
        if load_data_from == 'middata':
            kwargs = cls.load_data(cfg)
        elif load_data_from == 'rawdata':
            r2m_op = cls._get_r2m_op(cfg)
            r2m_op.process() # 生成middata文件夹
            kwargs = cls.load_data(cfg) # 基于middata文件夹导入数据
        else:
            kwargs = {}

        return cls(cfg, **kwargs)
    
    @property
    def common_str2df(self):
        """get the common data object

        Returns:
            dict: the common data object through the atomic operations
        """
        return {
            "df": self.df, "df_train": self.df_train, "df_valid": self.df_valid,
            "df_test": self.df_test, "dt_info": self.datatpl_cfg['dt_info']
        }
    
    def process_load_data_from_middata(self):
        """process middata
        """
        kwargs = self.common_str2df
        for op in self.m2c_op_list:
            kwargs = op.process(**kwargs)
            assert kwargs is not None
            op.set_dt_info(**kwargs)
        
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.final_kwargs = kwargs

        if self.datatpl_cfg['is_dataset_divided']:
            raise NotImplementedError # 统计数据集

    def process_data(self):
        """process data
        """
        load_data_from = self.datatpl_cfg['load_data_from']
        if load_data_from != 'cachedata':
            self.process_load_data_from_middata()
    
    @classmethod
    def load_data(cls, cfg): # 只在middata存在时调用
        """load data from disk
        """
        is_dataset_divided = cfg.datatpl_cfg['is_dataset_divided']
        if cfg.datatpl_cfg['n_folds'] == 1:
            kwargs = cls.load_data_from_undivided(cfg) # 1折时，如果数据没划分好，那么直接导入
        else:
            if is_dataset_divided: # 五折时，不支持已经划分好的
                raise ValueError("In the setting of n_fold>1, unsupport is_dataset_divided=True")
            else:
                kwargs = cls.load_data_from_undivided(cfg)
        for df in kwargs.values(): cls._preprocess_feat(df) # 类型转换
        return kwargs

    @classmethod
    def load_data_from_undivided(cls, cfg):
        """load undivided data from disk
        """
        fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.inter.csv'
        
        exclude_feats = cfg.datatpl_cfg['inter_exclude_feat_names']
        assert len(set(exclude_feats) & {'stu_id:token', 'exer_id:token', 'label:float'}) == 0

        sep = cfg.datatpl_cfg['seperator']
        df = cls._load_atomic_csv(fph, exclude_headers=exclude_feats, sep=sep)

        return {"df": df}
    
    @classmethod
    def load_data_from_divided(cls, cfg):
        """load divided data from disk
        """
        train_fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.train.inter.csv'
        valid_fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.valid.inter.csv'
        test_fph = f'{cfg.frame_cfg.data_folder_path}/middata/{cfg.dataset}.test.inter.csv'

        exclude_feats = cfg.datatpl_cfg['inter_exclude_feat_names']
        assert len(set(exclude_feats) & {'stu_id:token', 'exer_id:token', 'label:float'}) == 0

        sep = cfg.datatpl_cfg['seperator']
        df_train = cls._load_atomic_csv(train_fph, exclude_headers=exclude_feats, sep=sep)
        df_test = cls._load_atomic_csv(test_fph, exclude_headers=exclude_feats, sep=sep)
        df_valid = None
        if os.path.exists(valid_fph):
            df_valid = cls._load_atomic_csv(valid_fph, exclude_headers=exclude_feats, sep=sep)

        return {"df_train": df_train, "df_valid": df_valid, "df_test": df_test}

    @staticmethod
    def _load_atomic_csv(fph, exclude_headers, sep=','):
        headers = pd.read_csv(fph, nrows=0).columns.tolist()
        df = pd.read_csv(fph, sep=sep, encoding='utf-8', usecols=set(headers) - set(exclude_headers))
        return df

    def build_dataloaders(self):
        """build dataloaders that would be convey to training template
        """
        batch_size = self.traintpl_cfg['batch_size']
        num_workers = self.traintpl_cfg['num_workers']
        eval_batch_size = self.traintpl_cfg['eval_batch_size']
        train_dt_list, valid_dt_list, test_dt_list = self.build_datasets()
        train_loader_list, valid_loader_list, test_loader_list = [], [], []

        for fid in range(self.datatpl_cfg['n_folds']):
            train_loader = DataLoader(dataset=train_dt_list[fid], shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
            train_loader_list.append(train_loader)
            if self.hasValidDataset:
                valid_loader = DataLoader(dataset=valid_dt_list[fid], shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
                valid_loader_list.append(valid_loader)
            test_loader = DataLoader(dataset=test_dt_list[fid], shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
            test_loader_list.append(test_loader)
        
        return train_loader_list, valid_loader_list, test_loader_list

    def save_cache(self):
        """save cache data
        """
        # chech path
        if os.path.exists(self.cache_folder_path):
            raise ValueError(f"cache_fold({self.cache_folder_path}) already exists, won't save cache")
        else:
            os.makedirs(self.cache_folder_path)

        # save cache
        train_folds_fph = f"{self.cache_folder_path}/dict_train_folds.pkl"
        valid_folds_fph = f"{self.cache_folder_path}/dict_valid_folds.pkl"
        test_folds_fph = f"{self.cache_folder_path}/dict_test_folds.pkl"
        final_kwargs_fph = f"{self.cache_folder_path}/final_kwargs.pkl"

        self.save_pickle(train_folds_fph, self.dict_train_folds)
        self.save_pickle(valid_folds_fph, self.dict_valid_folds)
        self.save_pickle(test_folds_fph, self.dict_test_folds)
        self.save_pickle(final_kwargs_fph, self.final_kwargs)

        with open(f"{self.cache_folder_path}/datatpl_cfg.json", 'w', encoding='utf-8') as f:
            json.dump(json.loads(self.datatpl_cfg.dump_fmt()), fp=f, indent=2, ensure_ascii=False)

    def check_cache(self):
        """check whether the cache data is consistent with current config
        """
        with open(f"{self.cache_folder_path}/datatpl_cfg.json", 'r', encoding='utf-8') as f:
            cache_datatpl_cfg = json.load(f)
        
        temp_cache_datatpl_cfg = copy.deepcopy(cache_datatpl_cfg)
        del temp_cache_datatpl_cfg['dt_info']
        del temp_cache_datatpl_cfg['load_data_from']
        # del temp_cache_datatpl_cfg['raw2mid_op']
        # del temp_cache_datatpl_cfg['mid2cache_op_seq']
        curr_datatpl_cfg = copy.deepcopy(json.loads(self.datatpl_cfg.dump_fmt()))
        del curr_datatpl_cfg['dt_info']
        del curr_datatpl_cfg['load_data_from']
        # del curr_datatpl_cfg['raw2mid_op']
        # del curr_datatpl_cfg['mid2cache_op_seq']
        diff = DeepDiff(temp_cache_datatpl_cfg, curr_datatpl_cfg)

        if len(diff) != 0:
            raise ValueError(f"check cache error: {diff}")
        
        self.datatpl_cfg['dt_info'] = cache_datatpl_cfg['dt_info'] # partial load_cache

    def load_cache(self):
        """load cache data from disk
        """
        if self.datatpl_cfg['load_data_from'] != 'cachedata':
            return
        if not os.path.exists(self.cache_folder_path):
            raise ValueError(f"cache_fold({self.cache_folder_path}) not exists, can't load cache")
        
        train_folds_fph = f"{self.cache_folder_path}/dict_train_folds.pkl"
        valid_folds_fph = f"{self.cache_folder_path}/dict_valid_folds.pkl"
        test_folds_fph = f"{self.cache_folder_path}/dict_test_folds.pkl"
        final_kwargs_fph = f"{self.cache_folder_path}/final_kwargs.pkl"

        self.dict_train_folds = self.load_pickle(train_folds_fph)
        self.dict_valid_folds = self.load_pickle(valid_folds_fph)
        self.dict_test_folds = self.load_pickle(test_folds_fph)
        self.final_kwargs = self.load_pickle(final_kwargs_fph)

    def build_datasets(self):
        """build datasets
        """
        n_folds = self.datatpl_cfg['n_folds']
        assert len(self.dict_train_folds) == n_folds
        assert self.status.mode == DataTPLMode.MANAGER

        train_dt_list, valid_dt_list, test_dt_list = [], [], []

        for fid in range(n_folds):
            train_dataset = self._copy()
            train_dataset.set_mode(DataTPLMode.TRAIN, fid)
            train_dt_list.append(train_dataset)

            valid_dataset = None
            if self.hasValidDataset:
                valid_dataset = self._copy()
                valid_dataset.set_mode(DataTPLMode.VALID, fid)
                valid_dt_list.append(valid_dataset)
            
            test_dataset = self._copy()
            test_dataset.set_mode(DataTPLMode.TEST, fid)
            test_dt_list.append(test_dataset)
        
        return train_dt_list, valid_dt_list, test_dt_list

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)
    
    def set_mode(self, mode: DataTPLMode, fold_id):
        """set mode of current data template

        Args:
            mode (DataTPLMode): mode
            fold_id (int): id of fold
        """
        self.status.mode = mode
        self.status.fold_id = fold_id
        if mode is DataTPLMode.MANAGER:
            self._set_mode_manager()
        elif mode is DataTPLMode.TRAIN:
            self._set_mode_train()
        elif mode is DataTPLMode.VALID:
            self._set_mode_valid()
        elif mode is DataTPLMode.TEST:
            self._set_mode_test()
        else:
            raise ValueError(f"unknown type of mode:{mode}")

        self.length = next(iter(self.dict_main.values())).shape[0]

    def _set_mode_manager(self):
        """progress of manager mode
        """
        self.dict_main = None
        self.status.mode = DataTPLMode.MANAGER
        self.status.fold_id = None

    def _set_mode_train(self):
        """progress of train mode
        """
        self.dict_main = self.dict_train_folds[self.status.fold_id]
    
    def _set_mode_valid(self):
        """progress of valid mode
        """
        self.dict_main = self.dict_valid_folds[self.status.fold_id]
    
    def _set_mode_test(self):
        """progress of test mode
        """
        self.dict_main = self.dict_test_folds[self.status.fold_id]

    def _check_params(self):
        """check validation of default parameters
        """
        super()._check_params()
        assert self.datatpl_cfg['load_data_from'] in {'rawdata', 'middata', 'cachedata'}
        assert 'dt_info' not in self.datatpl_cfg
        assert self.datatpl_cfg['n_folds'] > 0

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return {
            k: v[index] for k,v in self.dict_main.items()
        }
    
    def _copy(self):
        """copy current instance
        """
        obj = super()._copy()
        obj.status = copy.copy(obj.status)
        return obj
    
    @property
    def hasValidDataset(self):
        """whether exists validation dataset
        """
        return self.dict_valid_folds is not None and len(self.dict_valid_folds) > 0
    
    @property
    def cache_folder_path(self):
        """folder path of cache data
        """
        save_cache_id = self.datatpl_cfg['cache_id']
        return f"{self.frame_cfg.data_folder_path}/cachedata/{self.cfg.dataset}_{save_cache_id}/"
    
    def save_pickle(self, filepath, obj):
        """save data into pickle file
        """
        with open(filepath, 'wb') as fb:
            pickle.dump(obj, fb, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_pickle(self, filepath):
        """load data into pickle file
        """
        with open(filepath, 'rb') as fb:
            return pickle.load(fb)

    @classmethod
    def get_default_cfg(cls, mid2cache_op_seq, **kwargs):
        """Get final default config object

        Args:
            mid2cache_op_seq (List[Union[BaseMid2Cache,str]]): Mid2Cahce Sequence

        Returns:
            UnifyConfig: the final default config object
        """
        cfg = UnifyConfig()
        for _cls in cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)

        # 获取mid_cache_op_seq
        mid2cache_op_seq = None
        if mid2cache_op_seq is None:
            for _cls in cls.__mro__:
                if not hasattr(_cls, 'default_cfg'):
                    break
                mid2cache_op_seq = _cls.default_cfg.get('mid2cache_op_seq', None)
                if mid2cache_op_seq is not None: break
            assert mid2cache_op_seq is not None
        
        # cfg.mid2cache_op_seq = mid2cache_op_seq
        from edustudio.atom_op.mid2cache import BaseMid2Cache
        from edustudio.atom_op.raw2mid import BaseRaw2Mid
        for op in mid2cache_op_seq:
            if isinstance(op, str):
                op = importlib.import_module('edustudio.atom_op.mid2cache').__getattribute__(op)
            elif issubclass(op, BaseMid2Cache):
                pass
            else:
                raise ValueError(f"Unknown type of Mid2Cache: {op}")
            cfg[op.__name__] = op.get_default_cfg()

        return cfg

    @classmethod
    def _get_r2m_op(cls, cfg):
        """Get Raw2Mid operation

        Args:
            cfg (UnifyConfig): the global config object

        Returns:
            BaseRaw2Mid: Raw2Mid operation
        """
        from edustudio.atom_op.raw2mid import BaseRaw2Mid
        r2m_op = cfg.datatpl_cfg['raw2mid_op']
        assert r2m_op is not None or r2m_op != "None"
        if isinstance(r2m_op, str):
            r2m_op = importlib.import_module('edustudio.atom_op.raw2mid').__getattribute__(r2m_op)
        elif issubclass(r2m_op, BaseRaw2Mid):
            pass
        else:
            raise ValueError(f"unsupported raw2mid_op:{r2m_op}")
        return r2m_op.from_cfg(cfg)
    
    def _get_m2c_op_list(self):
        """Get Mid2Cache operation sequence

        Returns:
           List[BaseMid2Cache]: Mid2Cache operation sequence
        """
        from edustudio.atom_op.mid2cache import BaseMid2Cache
        m2c_op_list = self.datatpl_cfg['mid2cache_op_seq']
        op_list = []
        for op in m2c_op_list:
            if isinstance(op, str):
                op = importlib.import_module('edustudio.atom_op.mid2cache').__getattribute__(op)
            elif issubclass(op, BaseMid2Cache):
                pass
            else:
                raise ValueError(f"unsupported mid2cache_op:{op}")
            op_list.append(op.from_cfg(self.cfg))
        return op_list


    def df2dict(self):
        """convert dataframe object into dict
        """
        # if self.datatpl_cfg['is_dataset_divided'] is True:
        #     self.df_train_folds.append(self.df_train)
        #     self.df_valid_folds.append(self.df_valid)
        #     self.df_test_folds.append(self.df_test)
        
        for tmp_df in self.df_train_folds:
            self.dict_train_folds.append(self._df2dict(tmp_df))
            
        if len(self.df_valid_folds) > 0:
            for tmp_df in self.df_valid_folds:
                self.dict_valid_folds.append(self._df2dict(tmp_df))
            
        for tmp_df in self.df_test_folds:
            self.dict_test_folds.append(self._df2dict(tmp_df))

    @staticmethod
    def _df2dict(dic_raw):
        """convert dataframe into dict
        """
        dic = {}
        for k in dic_raw:
            if type(dic_raw) is not dict:
                v = torch.from_numpy(dic_raw[k].to_numpy())
            else:
                v = dic_raw[k]
            if ":" in k:
                k = k.split(":")[0]
            dic[k] = v
        return dic

    def get_extra_data(self, **kwargs):
        """an interface to construct extra data except the data from forward API
        """
        extra_data = super().get_extra_data(**kwargs)
        # extra_data.update({
        #     'df': self.df,
        #     'df_train': self.df_train,
        #     'df_valid': self.df_valid,
        #     'df_test': self.df_test
        # })
        extra_data.update(self.final_kwargs)
        return extra_data
    
    @staticmethod
    def _preprocess_feat(df):
        """convert data format after loading files according to field type

        Args:
            df (DataFrame): data
        """
        for col in df.columns:
            col_name, col_type = col.split(":")
            if col_type == 'token':
                try:
                    df[col] = df[col].astype('int64')
                except:
                    pass
            elif col_type == 'float':
                df[col] = df[col].astype('float32')
            elif col_type == 'token_seq':
                try:
                    df[col] = df[col].astype(str).apply(lambda x: [int(i) for i in x.split(",")])
                except:
                    df[col] = df[col].astype(str).apply(lambda x: eval(x))
            elif col_type == 'float_seq':
                try:
                    df[col] = df[col].astype(str).apply(lambda x: [float(i) for i in x.split(",")])
                except:
                    df[col] = df[col].astype(str).apply(lambda x: eval(x))
            else:
                pass
            
    @staticmethod
    def _unwrap_feat(df:pd.DataFrame):
        """unwrap the type of field

        Args:
            df (pd.DataFrame): dataframe after unwrapping
        """
        for col in df.columns:
            col_name, col_type = col.split(":")
            df.rename(columns={col:col_name}, inplace=True)
        
    @classmethod
    def get_default_cfg(cls, **kwargs):
        """Get the final default config object

        Returns:
            UnifyConfig: the final default config object
        """
        cfg = UnifyConfig()
        for _cls in cls.__mro__:
            if not hasattr(_cls, 'default_cfg'):
                break
            cfg.update(_cls.default_cfg, update_unknown_key_only=True)
        
        for op in kwargs.get('mid2cache_op_seq', None) or cls.default_cfg['mid2cache_op_seq']:
            if isinstance(op, str):
                op = importlib.import_module('edustudio.atom_op.mid2cache').__getattribute__(op)
            if op.__name__ not in cfg:
                cfg[op.__name__] = op.get_default_cfg()
            else:
                cfg[op.__name__] = UnifyConfig(cfg[op.__name__])
                cfg[op.__name__].update(op.get_default_cfg(), update_unknown_key_only=True)
        return cfg

    
    def set_info_for_fold(self, fold_id):
        """Get data information when a specifying fold id

        Args:
            fold_id (int): id of fold
        """
        pass
