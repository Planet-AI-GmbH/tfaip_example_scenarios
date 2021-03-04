from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip.base import ScenarioBaseParams
from tfaip.base.scenario.scenariobase import ScenarioBase
from tfaip.base.trainer.scheduler import WarmupCosineDecay

from tfaipscenarios.text.finetuningbert.data import FTBertData
from tfaipscenarios.text.finetuningbert.datapipeline.gluedata import GlueTrainerPipelineParams
from tfaipscenarios.text.finetuningbert.datapipeline.tokenizerprocessor import TokenizerProcessorParams
from tfaipscenarios.text.finetuningbert.model import FTBertModelParams, FTBertModel
from tfaipscenarios.text.finetuningbert.params import FTBertDataParams


@pai_dataclass
@dataclass
class FTBertScenarioParams(ScenarioBaseParams[FTBertDataParams, FTBertModelParams]):
    def __post_init__(self):
        for p in self.data.pre_proc.processors_of_type(TokenizerProcessorParams):
            p.gs_folder_bert = self.model.gs_folder_bert


class FTBertScenario(ScenarioBase[FTBertData, FTBertModel, FTBertScenarioParams, GlueTrainerPipelineParams]):
    @classmethod
    def default_trainer_params(cls):
        p = super().default_trainer_params()
        p.gen.setup.train.batch_size = 32
        p.gen.setup.val.batch_size = 32
        p.epochs = 3
        p.learning_rate = WarmupCosineDecay(lr=2e-5, warmup_epochs=1, warmup_factor=100)
        return p
