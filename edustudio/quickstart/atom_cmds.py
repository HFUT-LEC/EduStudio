import fire
from edustudio.atom_op.raw2mid import _cli_api_dict_ as raw2mid
from edustudio.utils.common import Logger

def entrypoint():
    Logger().get_std_logger()
    fire.Fire(
        {"r2m": raw2mid}
    )
