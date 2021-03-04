from dataclasses import dataclass

from paiargparse import pai_dataclass
from tfaip.base import Sample
from tfaip.base.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor

from tfaipscenarios.imageclassification.params import Keys


@pai_dataclass
@dataclass
class IndexToClassProcessorParams(DataProcessorParams):
    @staticmethod
    def cls():
        return IndexToClassProcessor


class IndexToClassProcessor(MappingDataProcessor):
    def apply(self, sample: Sample) -> Sample:
        sample.outputs[Keys.OutputClassName] = self.data_params.classes[sample.outputs[Keys.OutputClass]]
        return sample
