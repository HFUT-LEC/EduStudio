from typing import Dict, Any
import edustudio.settings as settings
from edustudio.utils.common import UnifyConfig
import argparse
from ast import literal_eval
from collections import defaultdict
import importlib

def get_global_cfg(
    dataset:str,
    cfg_file_name:str,
    trainfmt_cfg_dict: Dict[str, Any],
    datafmt_cfg_dict:  Dict[str, Any],
    evalfmt_cfg_dict:  Dict[str, Any],
    model_cfg_dict: Dict[str, Any],
    frame_cfg_dict:  Dict[str, Any],
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dt', type=str,
                        help='dataset name', dest='dataset', default=dataset)
    parser.add_argument('--cfg_file_name', '-f', type=str,
                        help='config filename', dest='cfg_file_name', default=cfg_file_name)
    parser.add_argument('--trainfmt_cfg.cls', '-trainfmt_cls', type=str,
                        dest='trainfmt_cls', default=trainfmt_cfg_dict.get('cls', None))
    parser.add_argument('--datafmt_cfg.cls', '-datafmt_cls', type=str,
                        dest='datafmt_cls', default=datafmt_cfg_dict.get('cls', None))
    parser.add_argument('--model_cfg.cls', '-model_cls', type=str,
                        dest='model_cls', default=model_cfg_dict.get('cls', None))
    parser.add_argument('--evalfmt_cfg.clses', '-evalfmt_clses', nargs='+',
                        dest='evalfmt_clses', default=evalfmt_cfg_dict.get('clses', None))
    parser.add_argument('--datafmt_cfg.backbone_datafmt_cls', '-datafmt_backbone_cls', type=str,
                        dest='backbone_datafmt_cls', default=datafmt_cfg_dict.get('backbone_datafmt_cls', None))
    parser.add_argument('--model_cfg.backbone_model_cls', '-model_backbone_cls', type=str,
                        dest='backbone_model_cls', default=model_cfg_dict.get('backbone_model_cls', None))
    args, unknown_args = parser.parse_known_args()
    assert args.dataset is not None
    
    unknown_arg_dict = defaultdict(dict)
    if len(unknown_args) > 0:
        assert len(unknown_args) % 2 == 0, \
            "number of extra parameters[except dt and cfg_file_name] from command line should be even"
        for i in range(int(len(unknown_args) / 2)):
            assert unknown_args[2 * i].startswith("--"), \
                "the first term in extra parameter[except dt and cfg_file_name] pair should start with '--'"
            key, value = unknown_args[2 * i][2:], unknown_args[2 * i + 1]
            if key.startswith('trainfmt_cfg.'):
                unknown_arg_dict['trainfmt_cfg'][key] = value
            elif key.startswith('datafmt_cfg.'):
                unknown_arg_dict['datafmt_cfg'][key] = value
            elif key.startswith('model_cfg.'):
                unknown_arg_dict['model_cfg'][key] = value
            elif key.startswith('evalfmt_cfg.'):
               unknown_arg_dict['evalfmt_cfg'][key] = value
            elif key.startswith('frame_cfg.'):
                unknown_arg_dict['frame_cfg'][key] = value
            else:
                raise ValueError(f"unsupported key: {key}")


    cfg = UnifyConfig({
        'trainfmt_cfg': UnifyConfig(), 'datafmt_cfg': UnifyConfig(),
        'model_cfg': UnifyConfig(), 'evalfmt_cfg': UnifyConfig(), 
        'frame_cfg': UnifyConfig()
    })
    cfg.dataset = args.dataset
        
    # process frame cfg
    cfg.frame_cfg = UnifyConfig.from_py_module(settings)
    for k,v in frame_cfg_dict.items():
        assert k in cfg.frame_cfg
        assert type(cfg.frame_cfg[k]) is type(v)
        cfg.frame_cfg[k] = v
    for k,v in unknown_arg_dict['frame_cfg'].items():
        assert k in cfg.frame_cfg
        if type(cfg.frame_cfg[k]) is not str:
            v = type(cfg.frame_cfg)(literal_eval(v))
        cfg.dot_set(k, v)

    trainfmt_cls = args.trainfmt_cls
    datafmt_cls = args.datafmt_cls
    model_cls = args.model_cls
    evalfmt_clses = args.evalfmt_clses
    if args.cfg_file_name is not None:
        yaml_cfg = UnifyConfig.from_yml_file(
            f"{cfg.frame_cfg['CFG_FOLDER_PATH']}/{cfg.dataset}/{args.cfg_file_name}"
        )
        assert 'frame_cfg' not in yaml_cfg
        assert 'dataset' not in yaml_cfg
        assert 'logger' not in yaml_cfg
        trainfmt_cls = trainfmt_cls or yaml_cfg.get('trainfmt_cfg', {'cls': None}).get("cls")
        datafmt_cls = datafmt_cls or yaml_cfg.get('datafmt_cfg', {'cls': None}).get("cls")
        model_cls = model_cls or yaml_cfg.get('model_cfg', {'cls': None}).get("cls")
        evalfmt_clses = evalfmt_clses or yaml_cfg.get('evalfmt_cfg', {'clses': None}).get("clses")

    assert trainfmt_cls is not None
    assert datafmt_cls is not None
    assert model_cls is not None
    assert evalfmt_clses is not None
    assert len(set(evalfmt_clses)) == len(evalfmt_clses)


    cfg.dot_set('trainfmt_cfg.cls', trainfmt_cls)
    cfg.dot_set('datafmt_cfg.cls', datafmt_cls)
    cfg.dot_set('model_cfg.cls', model_cls)
    cfg.dot_set('evalfmt_cfg.clses', evalfmt_clses)
    
    if isinstance(trainfmt_cls, str):
        cfg.trainfmt_cfg.update(
            importlib.import_module("edustudio.trainfmt").
            __getattribute__(trainfmt_cls).get_default_cfg()
        )
    else:
        cfg.trainfmt_cfg.update(trainfmt_cls.get_default_cfg())
    
    if isinstance(model_cls, str):
        cfg.model_cfg.update(
            importlib.import_module("edustudio.model").
            __getattribute__(model_cls).get_default_cfg(backbone_model_cls=args.backbone_model_cls)
        )
    else:
        cfg.model_cfg.update(model_cls.get_default_cfg(backbone_model_cls=args.backbone_model_cls))
    
    if isinstance(datafmt_cls, str):
        cfg.datafmt_cfg.update(
            importlib.import_module("edustudio.datafmt").
            __getattribute__(datafmt_cls).get_default_cfg(backbone_datafmt_cls=args.backbone_datafmt_cls)
        )
    else:
        cfg.datafmt_cfg.update(datafmt_cls.get_default_cfg(backbone_datafmt_cls=args.backbone_datafmt_cls))
    
    for evalfmt_cls in evalfmt_clses:
        if isinstance(evalfmt_cls, str):
            cfg.evalfmt_cfg[evalfmt_cls] = importlib.import_module("edustudio.evalfmt").\
                __getattribute__(evalfmt_cls).get_default_cfg()
        else:
            cfg.evalfmt_cfg[evalfmt_cls.__name__] = evalfmt_cls.get_default_cfg()

    # config file
    if args.cfg_file_name is not None:
        for config_name in ['trainfmt_cfg', 'datafmt_cfg', 'model_cfg']:
            for k,v in yaml_cfg[config_name].items():
                assert k in cfg[config_name], f"invalid key: {k}"
                if k == 'cls': continue
                assert type(cfg[config_name][k]) is type(v)
                cfg[config_name][k] = v
        for k,v in yaml_cfg['evalfmt_cfg'].items():
            if k == 'clses': continue
            assert k in cfg.evalfmt_cfg['clses'], f"invalid key: {k}"
            assert type(v) is dict
            for kk, vv in v.items():
                assert kk in cfg.evalfmt_cfg[k], f"invalid key: {kk}"
                assert type(cfg.evalfmt_cfg[k][kk]) is type(vv)
                cfg.evalfmt_cfg[k][kk] = vv

    # parameter dict
    for k,v in trainfmt_cfg_dict.items():
        if k == 'cls': continue
        assert k in cfg['trainfmt_cfg'], f"invalid key: {k}"
        assert type(cfg['trainfmt_cfg'][k]) is type(v)
        cfg['trainfmt_cfg'][k] = v
    for k,v in datafmt_cfg_dict.items():
        if k == 'cls' or k == 'backbone_datafmt_cls': continue
        assert k in cfg['datafmt_cfg'], f"invalid key: {k}"
        assert type(cfg['datafmt_cfg'][k]) is type(v)
        cfg['datafmt_cfg'][k] = v
    for k,v in model_cfg_dict.items():
        if k == 'cls' or k == 'backbone_model_cls': continue
        assert k in cfg['model_cfg'], f"invalid key: {k}"
        assert type(cfg['model_cfg'][k]) is type(v)
        cfg['model_cfg'][k] = v

    evalfmt_clses_name = [cls_ if isinstance(cls_, str) else cls_.__name__ for cls_ in cfg.evalfmt_cfg['clses']]
    for k,v in evalfmt_cfg_dict.items():
        if k == 'clses': continue
        assert k in evalfmt_clses_name, f"invalid key: {k}"
        assert type(v) is dict
        for kk, vv in v.items():
            assert kk in cfg.evalfmt_cfg[k], f"invalid key: {kk}"
            assert type(cfg.evalfmt_cfg[k][kk]) is type(vv)
            cfg.evalfmt_cfg[k][kk] = vv
    

    # command line
    for config_name in ['trainfmt_cfg', 'datafmt_cfg', 'model_cfg']:
        for k,v in unknown_arg_dict[config_name].items():
            if k == 'cls': continue
            kk = k.split('.')[-1]
            assert kk in cfg[config_name], f"invalid key: {kk}"
            if type(cfg[config_name][kk]) is not str:
                v = type(cfg[config_name][kk])(literal_eval(v))
            cfg.dot_set(k, v)
    for k,v in unknown_arg_dict["evalfmt_cfg"].items():
        kk = k.split('.')[-1]
        assert kk in cfg['evalfmt_cfg'], f"invalid key: {kk}"
        if kk == 'clses': continue
        if type(cfg.dot_get(k)) is not str:
            v = type(cfg.dot_get(k))(literal_eval(v))
        cfg.dot_set(k, v)
    
    return cfg
