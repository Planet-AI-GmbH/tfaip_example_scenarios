import logging

import tensorflow as tf
from tfaip.base.model.graphbase import GraphBase
from transformers import TFAutoModel

from tfaipscenarios.text.finetuningbert.model import FTBertModelParams
from tfaipscenarios.text.finetuningbert.params import Keys

logger = logging.getLogger(__name__)


class FTBertGraph(GraphBase[FTBertModelParams]):
    def __init__(self, params: FTBertModelParams, **kwargs):
        super(FTBertGraph, self).__init__(params, **kwargs)
        self.bert = TFAutoModel.from_pretrained(params.model_name)
        self.logits = tf.keras.layers.Dense(2)

    def call(self, inputs, **kwargs):
        logits = self.logits(self.bert(inputs, **kwargs).pooler_output)
        return {
            Keys.OutputLogits: logits,
            Keys.OutputSoftmax: tf.nn.softmax(logits),
            Keys.OutputClass: tf.argmax(logits, axis=-1),
        }
