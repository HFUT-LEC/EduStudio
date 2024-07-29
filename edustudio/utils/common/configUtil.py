# -*- coding: utf-8 -*-
# @Author : Xiangzhi Chen
# @Github : kervias

import copy
from typing import Union, Any, Optional
import os
import json
import yaml
import types
import re
from functools import reduce
import numpy as np


class UnifyConfig(object):
    """
        Unified config object
    """

    def __init__(self, dic: Optional[dict] = None):
        self.__config__ = dic or dict()

    @classmethod
    def from_py_module(cls, module_object: types.ModuleType):
        return cls(
            {k: (getattr(module_object, k) if not k.endswith("_PATH") else os.path.realpath(getattr(module_object, k)))
             for k in dir(module_object) if k[0].isupper()}
        )

    @classmethod
    def from_yml_file(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=cls._build_yaml_loader())
        return cls(config or dict())

    @staticmethod
    def _build_yaml_loader():
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X
            ), list(u'-+0123456789.')
        )
        return loader

    def dot_contains(self, dot_string: str):
        keys = [self.__config__] + dot_string.strip().split('.')
        try:
            reduce(lambda x, y: x[y], keys)
        except Exception:
            return False
        return True

    def dot_get(self, dot_string: str, default_value: Any = None, require=False):
        keys = [self.__config__] + dot_string.strip().split('.')
        if not require:
            try:
                return reduce(lambda x, y: x[y], keys)
            except KeyError:
                return default_value
        else:
            return reduce(lambda x, y: x[y], keys)

    def dot_set(self, dot_string: str, value: Any = None):
        keys = dot_string.strip().split('.')
        obj = reduce(lambda x, y: x[y], [self.__config__] + keys[:-1])
        obj[keys[-1]] = value

    def __iter__(self):
        for k in self.__config__.keys():
            yield k

    def __getattr__(self, key: str):
        if '__config__' in self.__dict__ and key in self.__dict__['__config__']:
            return self.__dict__['__config__'][key]
        elif key in dir(self):
            return self.__dict__[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if key != "__config__":
            self.__config__[key] = value
        else:
            self.__dict__[key] = value

    def __delattr__(self, key: str):
        if key in self.__config__.keys():
            del self.__config__[key]
        elif key in dir(self):
            raise AttributeError(f"attribute '{key}' is not allowed to delete")
        else:
            raise AttributeError(f"'{self.__class__.__base__}' object has no attribute '{key}'")

    def __setitem__(self, key: str, value: Any):
        assert key not in dir(self), "conflict with dir(self)"
        self.__config__[key] = value

    def __getitem__(self, key: str):
        return self.__config__[key]

    def __delitem__(self, key: str):
        del self.__config__[key]

    def keys(self):
        return self.__config__.keys()

    def items(self):
        return self.__config__.items()

    def to_dict(self):
        return copy.deepcopy(self.__config__)

    def get(self, key: str, default_value: Any = None):
        return self.__config__.get(key, default_value)

    def update(self, dict_obj: Union[dict, object], update_unknown_key_only=False):
        for k in dict_obj:
            if k in self and update_unknown_key_only:
                continue
            self[k] = dict_obj[k]

    def __str__(self):
        return f"{self.__class__.__name__}({self.__config__})"

    def __repr__(self):
        return self.__str__()

    def dump_fmt(self):
        return json.dumps(
            self.__config__, indent=4, ensure_ascii=False,
            cls=NumpyEncoder
        )

    def dump_file(self, filepath: str, encoding: str = 'utf-8'):
        with open(filepath, "w", encoding=encoding) as f:
            json.dump(
                self.__config__, f, indent=4, ensure_ascii=False,
                cls=NumpyEncoder
            )

    def __copy__(self):
        cls = self.__class__
        return cls(dic=copy.copy(self.__config__))

    def __deepcopy__(self, memo: Any):
        cls = self.__class__
        return cls(dic=copy.deepcopy(self.__config__, memo=memo))


class NumpyEncoder(json.JSONEncoder):
    """ 
    Custom encoder for numpy data types 
    Ref: https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
    """

    def default(self, obj):
        if isinstance(obj, UnifyConfig):
            return obj.__config__
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)
