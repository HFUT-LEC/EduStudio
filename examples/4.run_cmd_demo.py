import fire
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))


from edustudio.atom_op.raw2mid import _cli_api_dict_ as raw2mid
from edustudio.utils.common import Logger

def entrypoint():
    Logger().get_std_logger()
    fire.Fire(
        {"r2m": raw2mid}
    )

entrypoint()
