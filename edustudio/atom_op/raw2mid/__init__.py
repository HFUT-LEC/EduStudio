from .raw2mid import BaseRaw2Mid
from .assist_0910 import R2M_ASSIST_0910
from .frcsub import R2M_FrcSub
from .assist_1213 import R2M_ASSIST_1213
from .math1 import R2M_Math1
from .math2 import R2M_Math2
from .aaai_2023 import R2M_AAAI_2023
from .algebra2005 import R2M_Algebra_0506
from .assist_1516 import R2M_ASSIST_1516
from .assist_2017 import R2M_ASSIST_17
from .bridge2006 import R2M_Bridge2Algebra_0607
from .ednet_kt1 import R2M_EdNet_KT1
from .junyi_area_topic_as_cpt import R2M_Junyi_AreaTopicAsCpt
from .junyi_exer_as_cpt import R2M_JunyiExerAsCpt
from .nips12 import R2M_Eedi_20_T12
from .nips34 import R2M_Eedi_20_T34
from .simulated5 import R2M_Simulated5
from .slp_english import R2M_SLP_English
from .slp_math import R2M_SLP_Math

# look up api dict
_cli_api_dict_ = {}
_cli_api_dict_['R2M_ASSIST_0910'] = R2M_ASSIST_0910.from_cli
_cli_api_dict_['R2M_FrcSub'] = R2M_FrcSub.from_cli
_cli_api_dict_['R2M_ASSIST_1213'] = R2M_ASSIST_1213.from_cli
_cli_api_dict_['R2M_Math1'] = R2M_Math1.from_cli
_cli_api_dict_['R2M_Math2'] = R2M_Math2.from_cli
_cli_api_dict_['R2M_Xueersi_2023'] = R2M_AAAI_2023.from_cli
_cli_api_dict_['R2M_Algebra_0506'] = R2M_Algebra_0506.from_cli
_cli_api_dict_['R2M_ASSIST_1516'] = R2M_ASSIST_1516.from_cli
_cli_api_dict_['R2M_ASSIST_17'] = R2M_ASSIST_17.from_cli
_cli_api_dict_['R2M_Bridge2Algebra_0607'] = R2M_Bridge2Algebra_0607.from_cli
_cli_api_dict_['R2M_EdNet_KT1'] = R2M_EdNet_KT1.from_cli
_cli_api_dict_['R2M_Junyi_AreaTopicAsCpt'] = R2M_Junyi_AreaTopicAsCpt.from_cli
_cli_api_dict_['R2M_JunyiExerAsCpt'] = R2M_JunyiExerAsCpt.from_cli
_cli_api_dict_['R2M_Eedi_20_T12'] = R2M_Eedi_20_T12.from_cli
_cli_api_dict_['R2M_Eedi_20_T34'] = R2M_Eedi_20_T34.from_cli
_cli_api_dict_['R2M_Simulated5'] = R2M_Simulated5.from_cli
_cli_api_dict_['R2M_SLP_Math'] = R2M_SLP_Math.from_cli
_cli_api_dict_['R2M_SLP_English'] = R2M_SLP_English.from_cli
