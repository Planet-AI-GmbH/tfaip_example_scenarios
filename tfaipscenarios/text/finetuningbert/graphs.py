import json
import os
import logging

import tensorflow as tf
from official.nlp.bert.bert_models import classifier_model
from official.nlp.bert.configs import BertConfig
from tfaip.base.model.graphbase import GraphBase

from tfaipscenarios.text.finetuningbert.model import FTBertModelParams
from tfaipscenarios.text.finetuningbert.params import Keys


logger = logging.getLogger(__name__)

class FTBertGraph(GraphBase[FTBertModelParams]):
    def __init__(self, params: FTBertModelParams, **kwargs):
        super(FTBertGraph, self).__init__(params, **kwargs)

        bert_config_file = os.path.join(params.gs_folder_bert, "bert_config.json")
        config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
        bert_config = BertConfig.from_dict(config_dict)
        self.bert_classifier, self.bert_encoder = classifier_model(bert_config, num_labels=2)
        checkpoint = tf.train.Checkpoint(model=self.bert_encoder)
        logger.info("Loading pretrained weights. This might take some time when called the first time...")
        checkpoint.restore(os.path.join(params.gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
        logger.info("Model restored.")

    def call(self, inputs, **kwargs):
        logits = self.bert_classifier(inputs, **kwargs)
        return {
            Keys.OutputLogits: logits,
            Keys.OutputSoftmax: tf.nn.softmax(logits),
            Keys.OutputClass: tf.argmax(logits, axis=-1),
        }
