from typing import Dict, Any
import edustudio.settings as settings
from edustudio.utils.common import IDUtil as idUtil
from edustudio.utils.common import UnifyConfig
import argparse
from ast import literal_eval
from collections import defaultdict
import importlib


def get_global_cfg(
    dataset:str,
    cfg_file_name:str,
    traintpl_cfg_dict: Dict[str, Any],
    datatpl_cfg_dict:  Dict[str, Any],
    evaltpl_cfg_dict:  Dict[str, Any],
    modeltpl_cfg_dict: Dict[str, Any],
    frame_cfg_dict:  Dict[str, Any],
):
    """merge configurations from different entrypoint into a global config object

    Args:
        dataset (str): dataset name
        cfg_file_name (str): config file name
        traintpl_cfg_dict (Dict[str, Any]): parameter dict of training template
        datatpl_cfg_dict (Dict[str, Any]): parameter dict of data template
        evaltpl_cfg_dict (Dict[str, Any]): parameter dict of evaluate template
        modeltpl_cfg_dict (Dict[str, Any]): parameter dict of model template
        frame_cfg_dict (Dict[str, Any]): parameter dict of framework template

    Returns:
        UnifyConfig: the global config object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dt', type=str,
                        help='dataset name', dest='dataset', default=dataset)
    parser.add_argument('--cfg_file_name', '-f', type=str,
                        help='config filename', dest='cfg_file_name', default=cfg_file_name)
    parser.add_argument('--traintpl_cfg.cls', '-traintpl_cls', type=str,
                        dest='traintpl_cls', default=traintpl_cfg_dict.get('cls', None))
    parser.add_argument('--datatpl_cfg.cls', '-datatpl_cls', type=str,
                        dest='datatpl_cls', default=datatpl_cfg_dict.get('cls', None))
    parser.add_argument('--modeltpl_cfg.cls', '-modeltpl_cls', type=str,
                        dest='modeltpl_cls', default=modeltpl_cfg_dict.get('cls', None))
    parser.add_argument('--evaltpl_cfg.clses', '-evaltpl_clses', nargs='+',
                        dest='evaltpl_clses', default=evaltpl_cfg_dict.get('clses', None))
    parser.add_argument('--datatpl_cfg.backbone_datatpl_cls', '-datatpl_backbone_cls', type=str,
                        dest='backbone_datatpl_cls', default=datatpl_cfg_dict.get('backbone_datatpl_cls', None))
    parser.add_argument('--modeltpl_cfg.backbone_modeltpl_cls', '-modeltpl_backbone_cls', type=str,
                        dest='backbone_modeltpl_cls', default=modeltpl_cfg_dict.get('backbone_modeltpl_cls', None))
    parser.add_argument('--datatpl_cfg.mid2cache_op_seq', '-mid2cache_op_seq', type=str,
                        dest='mid2cache_op_seq', default=datatpl_cfg_dict.get('mid2cache_op_seq', None))   

    try:
        __IPYTHON__
        _default_args = []
    except NameError:
        _default_args = None
    args, unknown_args = parser.parse_known_args(args=_default_args)
    assert args.dataset is not None
    
    unknown_arg_dict = defaultdict(dict)
    if len(unknown_args) > 0:
        assert len(unknown_args) % 2 == 0, \
            "number of extra parameters[except dt and cfg_file_name] from command line should be even"
        for i in range(int(len(unknown_args) / 2)):
            assert unknown_args[2 * i].startswith("--"), \
                "the first term in extra parameter[except dt and cfg_file_name] pair should start with '--'"
            key, value = unknown_args[2 * i][2:], unknown_args[2 * i + 1]
            if key.startswith('traintpl_cfg.'):
                unknown_arg_dict['traintpl_cfg'][key] = value
            elif key.startswith('datatpl_cfg.'):
                unknown_arg_dict['datatpl_cfg'][key] = value
            elif key.startswith('modeltpl_cfg.'):
                unknown_arg_dict['modeltpl_cfg'][key] = value
            elif key.startswith('evaltpl_cfg.'):
               unknown_arg_dict['evaltpl_cfg'][key] = value
            elif key.startswith('frame_cfg.'):
                unknown_arg_dict['frame_cfg'][key] = value
            else:
                pass
                # raise ValueError(f"unsupported key: {key}")


    cfg = UnifyConfig({
        'traintpl_cfg': UnifyConfig(), 'datatpl_cfg': UnifyConfig(),
        'modeltpl_cfg': UnifyConfig(), 'evaltpl_cfg': UnifyConfig(), 
        'frame_cfg': UnifyConfig()
    })
    cfg.dataset = args.dataset
        
    # process frame cfg
    cfg.frame_cfg = UnifyConfig.from_py_module(settings)
    cfg.frame_cfg.ID = idUtil.get_random_id_bytime()
    for k,v in frame_cfg_dict.items():
        assert k in cfg.frame_cfg
        assert type(v) is None or type(cfg.frame_cfg[k]) is type(v)
        cfg.frame_cfg[k] = v
    for k,v in unknown_arg_dict['frame_cfg'].items():
        assert k in cfg.frame_cfg
        if type(cfg.frame_cfg[k]) is not str:
            v = type(cfg.frame_cfg)(literal_eval(v))
        cfg.dot_set(k, v)

    traintpl_cls = args.traintpl_cls
    datatpl_cls = args.datatpl_cls
    modeltpl_cls = args.modeltpl_cls
    evaltpl_clses = args.evaltpl_clses
    if args.cfg_file_name is not None:
        yaml_cfg = UnifyConfig.from_yml_file(
            f"{cfg.frame_cfg['CFG_FOLDER_PATH']}/{cfg.dataset}/{args.cfg_file_name}"
        )
        assert 'frame_cfg' not in yaml_cfg
        assert 'dataset' not in yaml_cfg
        assert 'logger' not in yaml_cfg
        traintpl_cls = traintpl_cls or yaml_cfg.get('traintpl_cfg', {'cls': None}).get("cls")
        datatpl_cls = datatpl_cls or yaml_cfg.get('datatpl_cfg', {'cls': None}).get("cls")
        modeltpl_cls = modeltpl_cls or yaml_cfg.get('modeltpl_cfg', {'cls': None}).get("cls")
        evaltpl_clses = evaltpl_clses or yaml_cfg.get('evaltpl_cfg', {'clses': None}).get("clses")

        args.backbone_modeltpl_cls = args.backbone_modeltpl_cls or yaml_cfg.get('modeltpl_cfg', {'backbone_modeltpl_cls': None}).get("backbone_modeltpl_cls")
        args.backbone_datatpl_cls = args.backbone_datatpl_cls or yaml_cfg.get('datatpl_cfg', {'backbone_datatpl_cls': None}).get("backbone_datatpl_cls")
        args.mid2cache_op_seq = args.mid2cache_op_seq or yaml_cfg.get('datatpl_cfg', {'mid2cache_op_seq': None}).get("mid2cache_op_seq")

    assert traintpl_cls is not None
    assert datatpl_cls is not None
    assert modeltpl_cls is not None
    assert evaltpl_clses is not None
    assert len(set(evaltpl_clses)) == len(evaltpl_clses)


    cfg.dot_set('traintpl_cfg.cls', traintpl_cls)
    cfg.dot_set('datatpl_cfg.cls', datatpl_cls)
    cfg.dot_set('modeltpl_cfg.cls', modeltpl_cls)
    cfg.dot_set('evaltpl_cfg.clses', evaltpl_clses)
    
    if isinstance(traintpl_cls, str):
        cfg.traintpl_cfg.update(
            importlib.import_module("edustudio.traintpl").
            __getattribute__(traintpl_cls).get_default_cfg()
        )
    else:
        cfg.traintpl_cfg.update(traintpl_cls.get_default_cfg())
    
    if isinstance(modeltpl_cls, str):
        cfg.modeltpl_cfg.update(
            importlib.import_module("edustudio.model").
            __getattribute__(modeltpl_cls).get_default_cfg(backbone_modeltpl_cls=args.backbone_modeltpl_cls)
        )
    else:
        cfg.modeltpl_cfg.update(modeltpl_cls.get_default_cfg(backbone_modeltpl_cls=args.backbone_modeltpl_cls))
    
    if isinstance(datatpl_cls, str):
        cfg.datatpl_cfg.update(
            importlib.import_module("edustudio.datatpl").
            __getattribute__(datatpl_cls).get_default_cfg(backbone_datatpl_cls=args.backbone_datatpl_cls, mid2cache_op_seq=args.mid2cache_op_seq)
        )
    else:
        cfg.datatpl_cfg.update(datatpl_cls.get_default_cfg(backbone_datatpl_cls=args.backbone_datatpl_cls, mid2cache_op_seq=args.mid2cache_op_seq))
    
    for evaltpl_cls in evaltpl_clses:
        if isinstance(evaltpl_cls, str):
            cfg.evaltpl_cfg[evaltpl_cls] = importlib.import_module("edustudio.evaltpl").\
                __getattribute__(evaltpl_cls).get_default_cfg()
        else:
            cfg.evaltpl_cfg[evaltpl_cls.__name__] = evaltpl_cls.get_default_cfg()


    if args.mid2cache_op_seq is not None:
        atom_data_op_set = {(op if type(op) is str else op.__name__) for op in args.mid2cache_op_seq}
    else:
        atom_data_op_set = {(op if type(op) is str else op.__name__) for op in cfg['datatpl_cfg'].get('mid2cache_op_seq', [])}

    # config file
    if args.cfg_file_name is not None:
        for config_name in ['traintpl_cfg', 'datatpl_cfg', 'modeltpl_cfg', 'evaltpl_cfg']:
            if config_name not in yaml_cfg: continue
            if config_name == 'datatpl_cfg':
                for k,v in yaml_cfg[config_name].items():
                    assert k in cfg[config_name], f"invalid key: {k}"
                    if k == 'cls': continue
                    # assert type(v) is None or type(cfg[config_name][k]) is type(v)
                    if k in atom_data_op_set:
                        for kk,vv in yaml_cfg[config_name].get(k, {}).items():
                            assert kk in cfg[config_name][k], f"invalid key: {kk}"
                            cfg[config_name][k][kk] = vv
                    else:
                        cfg[config_name][k] = v 
            if config_name in ['traintpl_cfg', 'modeltpl_cfg']:
                for k,v in yaml_cfg[config_name].items():
                    assert k in cfg[config_name], f"invalid key: {k}"
                    if k == 'cls': continue
                    # assert type(v) is None or type(cfg[config_name][k]) is type(v)
                    cfg[config_name][k] = v
            if config_name in ['evaltpl_cfg']:
                for k,v in yaml_cfg[config_name].items():
                    if k == 'clses': continue
                    assert k in cfg.evaltpl_cfg['clses'], f"invalid key: {k}"
                    assert type(v) is dict
                    for kk, vv in v.items():
                        assert kk in cfg.evaltpl_cfg[k], f"invalid key: {kk}"
                        assert type(cfg.evaltpl_cfg[k][kk]) is type(vv)
                        cfg.evaltpl_cfg[k][kk] = vv

    # parameter dict
    for k,v in traintpl_cfg_dict.items():
        if k == 'cls': continue
        assert k in cfg['traintpl_cfg'], f"invalid key: {k}"
        # assert type(v) is None or type(cfg['traintpl_cfg'][k]) is type(v)
        cfg['traintpl_cfg'][k] = v
    
    for k,v in datatpl_cfg_dict.items():
        if k == 'cls' or k == 'backbone_datatpl_cls': continue
        assert k in cfg['datatpl_cfg'], f"invalid key: {k}"
        # assert type(v) is None or type(cfg['datatpl_cfg'][k]) is type(v)
        if k in atom_data_op_set:
            for kk,vv in datatpl_cfg_dict[k].items():
                assert kk in cfg['datatpl_cfg'][k], f"invalid key: {kk}"
                cfg['datatpl_cfg'][k][kk] = vv
        else:
            cfg['datatpl_cfg'][k] = v
    for k,v in modeltpl_cfg_dict.items():
        if k == 'cls' or k == 'backbone_modeltpl_cls': continue
        assert k in cfg['modeltpl_cfg'], f"invalid key: {k}"
        # assert type(v) is None or type(cfg['modeltpl_cfg'][k]) is type(v)
        cfg['modeltpl_cfg'][k] = v

    evaltpl_clses_name = [cls_ if isinstance(cls_, str) else cls_.__name__ for cls_ in cfg.evaltpl_cfg['clses']]
    for k,v in evaltpl_cfg_dict.items():
        if k == 'clses': continue
        assert k in evaltpl_clses_name, f"invalid key: {k}"
        assert type(v) is dict
        for kk, vv in v.items():
            assert kk in cfg.evaltpl_cfg[k], f"invalid key: {kk}"
            # assert type(cfg.evaltpl_cfg[k][kk]) is type(vv)
            cfg.evaltpl_cfg[k][kk] = vv
    

    # command line
    for config_name in ['traintpl_cfg', 'datatpl_cfg', 'modeltpl_cfg']:
        for k,v in unknown_arg_dict[config_name].items():
            if k == 'cls': continue
            kk = k.split('.')[-1]
            assert kk in cfg[config_name], f"invalid key: {kk}"
            if type(cfg[config_name][kk]) is not str:
                v = type(cfg[config_name][kk])(literal_eval(v))
            cfg.dot_set(k, v)
    for k,v in unknown_arg_dict["evaltpl_cfg"].items():
        kk = k.split('.')[-1]
        assert kk in cfg['evaltpl_cfg'], f"invalid key: {kk}"
        if kk == 'clses': continue
        if type(cfg.dot_get(k)) is not str:
            v = type(cfg.dot_get(k))(literal_eval(v))
        cfg.dot_set(k, v)
    
    return cfg
