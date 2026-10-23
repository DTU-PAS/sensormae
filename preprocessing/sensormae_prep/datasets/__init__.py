from .bdd import BDD100K
from .cityscapes import Cityscapes
from .dsec import DSEC
from .ecmd import ECMD
from .frt import FreiburgThermal
from .ms2 import MS2
from .nsavp import NSAVP
from .ntu4dradlm import NTU4DRadLM
from .rasmd import RASMD
from .sthereo import STheReO
from .thermal_pairs import FMB, KAIST, LLVIP, SMOD, MFNet
from .vividpp import VividPP

DATASETS = {d.name: d for d in (
    BDD100K, Cityscapes, DSEC, ECMD, FMB, FreiburgThermal, KAIST, LLVIP, MFNet, MS2,
    NSAVP, NTU4DRadLM, RASMD, SMOD, STheReO, VividPP,
)}
