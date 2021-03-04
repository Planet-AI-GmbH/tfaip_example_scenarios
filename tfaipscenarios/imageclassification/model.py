from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import tensorflow as tf
from paiargparse import pai_dataclass
from tfaip.base import ModelBaseParams
from tfaip.base.model.metric.simple import MetricDefinition
from tfaip.base.model.modelbase import ModelBase, TMP
from tfaip.util.typing import AnyNumpy

from tfaipscenarios.imageclassification.params import Keys


@pai_dataclass
@dataclass
class ICModelParams(ModelBaseParams):
    num_classes: int = -1
    conv_filters: List[int] = field(default_factory=lambda: [16, 32, 64])
    dense: List[int] = field(default_factory=lambda: [128])
    activation: str = 'relu'


class ICModel(ModelBase[ICModelParams]):
    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "acc"

    def create_graph(self, params: TMP):
        from tfaipscenarios.imageclassification.graphs import ICGraph
        return ICGraph(params)

    def _loss(self, inputs_targets: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def cross_entropy_wrapper(args):
            y_true, y_pred = args
            return tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)

        return {
            'loss': tf.keras.layers.Lambda(cross_entropy_wrapper, name='loss')(
                (inputs_targets[Keys.Target], outputs[Keys.OutputLogits]))
        }

    def _metric(self) -> Dict[str, MetricDefinition]:
        return {
            'acc': MetricDefinition(target=Keys.Target, output=Keys.OutputClass,
                                    metric=tf.keras.metrics.Accuracy(name='acc'))}

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data, print_fn):
        print_fn(f"TARGET/PREDICTION {targets[Keys.Target][0]}/{outputs[Keys.OutputClass]}")
