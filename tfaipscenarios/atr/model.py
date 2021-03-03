from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING, Any, Tuple

import Levenshtein
import tensorflow as tf
from paiargparse import pai_dataclass
from tfaip.base import ModelBaseParams
from tfaip.base.model.modelbase import ModelBase, TMP
from tfaip.util.typing import AnyNumpy

from tfaipscenarios.atr.params import Keys

if TYPE_CHECKING:
    from tfaipscenarios.atr.data import ATRData


@pai_dataclass
@dataclass
class ATRModelParams(ModelBaseParams):
    num_classes: int = -1
    conv_filters: List[int] = field(default_factory=lambda: [20, 40])
    lstm_nodes: int = 100
    dropout: float = 0.5


class ATRModel(ModelBase[ATRModelParams]):
    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "CER"

    def create_graph(self, params: TMP) -> 'GraphBase':
        from tfaipscenarios.atr.graphs import ATRGraph
        return ATRGraph(params)

    def _loss(self, inputs_targets: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return tf.keras.backend.expand_dims(tf.keras.backend.flatten(x), axis=-1)

        # note: blank is last index
        loss = tf.keras.layers.Lambda(
            lambda args: tf.keras.backend.ctc_batch_cost(args[0], args[1], args[2], args[3]), name='ctc')(
            (inputs_targets[Keys.Targets], outputs['blank_last_softmax'], to_2d_list(outputs['out_len']),
             to_2d_list(inputs_targets[Keys.TargetsLength])))
        return {
            'loss': loss
        }

    def _extended_metric(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def create_cer(decoded, targets, targets_length):
            # -1 is padding value which is expected to be 0, so shift all values by + 1
            greedy_decoded = tf.sparse.from_dense(decoded + 1)
            sparse_targets = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(targets + 1, tf.cast(
                tf.keras.backend.flatten(targets_length), dtype='int32')), 'int32')
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        cer = tf.keras.layers.Lambda(lambda args: create_cer(*args), output_shape=(1,), name='cer')(
            (outputs['decoded'], inputs[Keys.Targets], inputs[Keys.TargetsLength]))
        return {
            'CER': cer,
        }

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        return {
            "CER": tf.keras.backend.flatten(targets[Keys.TargetsLength]),
        }

    def print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                       data: 'ATRData', print_fn=print):
        # trim the sentences, decode them , compute their CER, amd print
        pred_sentence = ''.join([data.params.codec[i] for i in outputs['decoded'] if i >= 0])  # -1 is padding
        gt_sentence = ''.join([data.params.codec[i] for i in targets[Keys.Targets][:targets[Keys.TargetsLength][0]]])
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        print_fn(f"\n  CER:  {cer}" +
                 f"\n  PRED: '{pred_sentence}'" +
                 f"\n  TRUE: '{gt_sentence}'")
