#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .dedafp import DedaFp, DedaFpConfig
from .doraemon import Doraemon, DoraemonConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .ernie import Ernie, ErnieConfig
from .happo import Happo, HappoConfig
from .hasac import Hasac, HasacConfig
from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .ipga import Ipga, IpgaConfig
from .iql import Iql, IqlConfig
from .isac import Isac, IsacConfig
from .lcpo import Lcpo, LcpoConfig
from .liam import Liam, LiamConfig
from .m3w import M3w, M3wConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .mappo_ctde import MappoCtde, MappoCtdeConfig
from .masac import Masac, MasacConfig
from .mfac import Mfac, MfacConfig
from .qcd import Qcd, QcdConfig
from .qmix import Qmix, QmixConfig
from .rma import Rma, RmaConfig
from .vdn import Vdn, VdnConfig
from .wisdom import Wisdom, WisdomConfig

classes = [
    "DedaFp",
    "DedaFpConfig",
    "Doraemon",
    "DoraemonConfig",
    "Ernie",
    "ErnieConfig",
    "Happo",
    "HappoConfig",
    "Hasac",
    "HasacConfig",
    "Iddpg",
    "IddpgConfig",
    "Ipga",
    "IpgaConfig",
    "Ippo",
    "IppoConfig",
    "Iql",
    "IqlConfig",
    "Isac",
    "IsacConfig",
    "Lcpo",
    "LcpoConfig",
    "Liam",
    "LiamConfig",
    "M3w",
    "M3wConfig",
    "Maddpg",
    "MaddpgConfig",
    "Mappo",
    "MappoConfig",
    "MappoCtde",
    "MappoCtdeConfig",
    "Masac",
    "MasacConfig",
    "Mfac",
    "MfacConfig",
    "Qcd",
    "QcdConfig",
    "Qmix",
    "QmixConfig",
    "Rma",
    "RmaConfig",
    "Vdn",
    "VdnConfig",
    "Wisdom",
    "WisdomConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "mappo": MappoConfig,
    # ---- BASELINES.md baselines; see baselines/README.md -------------
    "happo": HappoConfig,      # B1  trust region / sequential update
    "hasac": HasacConfig,      # B1  off-policy, maximum entropy
    "ernie": ErnieConfig,      # B9  robust MARL, adversarial regulariser
    "lcpo": LcpoConfig,        # B6  non-stationary RL, observed context
    "liam": LiamConfig,        # B5  agent modelling
    "mfac": MfacConfig,        # B4  mean-field MARL
    "rma": RmaConfig,          # B8  meta-RL / online system identification
    # ---- the EXTRA baselines; see baselines/README_EXTRA.md ----------
    "qcd": QcdConfig,          # X1  prior-free NS-RL: detect and restart
    "dedafp": DedaFpConfig,    # X2  deep fictitious play for continuous MFGs
    "ipga": IpgaConfig,        # X3  independent learning, performative MPGs
    "wisdom": WisdomConfig,    # X4  wavelet predictive representations
    "doraemon": DoraemonConfig,  # X5  DR via entropy maximisation
    "m3w": M3wConfig,          # X6  MoE world model, with planning
    "mappo_ctde": MappoCtdeConfig,
    "ippo": IppoConfig,
    "maddpg": MaddpgConfig,
    "iddpg": IddpgConfig,
    "masac": MasacConfig,
    "isac": IsacConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
}
