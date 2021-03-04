from typing import Dict

import tensorflow as tf
from tfaip.base.data.data import DataBase

from tfaipscenarios.imageclassification.datapipeline.indextoclassprocessor import IndexToClassProcessorParams
from tfaipscenarios.imageclassification.datapipeline.loadprocessor import LoadProcessorParams
from tfaipscenarios.imageclassification.datapipeline.preparesampleprocessor import PrepareSampleProcessorParams
from tfaipscenarios.imageclassification.datapipeline.rescaleprocessor import RescaleProcessorParams
from tfaipscenarios.imageclassification.params import Keys, ICDataParams


class ICData(DataBase[ICDataParams]):
    @classmethod
    def default_params(cls) -> ICDataParams:
        p = super().default_params()
        p.pre_proc.processors = [
            LoadProcessorParams(),
            RescaleProcessorParams(),
            PrepareSampleProcessorParams(),
        ]
        p.post_proc.run_parallel = False
        p.post_proc.processors = [
            IndexToClassProcessorParams()
        ]
        return p

    def _input_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Image: tf.TensorSpec([self.params.image_height, self.params.image_width, 3], tf.uint8),
        }

    def _target_layer_specs(self) -> Dict[str, tf.TensorSpec]:
        return {
            Keys.Target: tf.TensorSpec([1], tf.int32),
        }
