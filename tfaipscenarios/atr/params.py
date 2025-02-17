import os
from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip.base import DataBaseParams, TrainerPipelineParams

from tfaipscenarios.atr.datapipeline.datagenerator import ATRDataGeneratorParams

this_dir = os.path.dirname(os.path.realpath(__file__))


class Keys:
    Targets = 'targets'
    TargetsLength = 'targets_length'
    Image = 'image'
    ImageLength = 'imageLength'


def read_default_codec() -> str:
    with open(os.path.join(this_dir, 'workingdir', 'uw3_50lines', 'full_codec.txt')) as f:
        return f.read()


@pai_dataclass
@dataclass
class ATRDataParams(DataBaseParams):
    codec: str = read_default_codec()
    height: int = 48


# Setup the training pipeline to use both ATRDataGeneratorParams for the training- and validation-set
@pai_dataclass
@dataclass
class ATRTrainerPipelineParams(TrainerPipelineParams[ATRDataGeneratorParams, ATRDataGeneratorParams]):
    pass
