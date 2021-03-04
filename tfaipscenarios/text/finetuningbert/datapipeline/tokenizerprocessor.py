import logging
import os
from dataclasses import dataclass
from typing import Type

import numpy as np
from paiargparse import pai_dataclass
from tfaip.base import Sample
from tfaip.base.data.pipeline.processor.dataprocessor import DataProcessorParams, MappingDataProcessor

from tfaipscenarios.text.finetuningbert.params import Keys

logger = logging.getLogger(__name__)


@pai_dataclass
@dataclass
class TokenizerProcessorParams(DataProcessorParams):
    gs_folder_bert: str = ''

    @staticmethod
    def cls() -> Type['DataProcessorBase']:
        return TokenizerProcessor


class TokenizerProcessor(MappingDataProcessor[TokenizerProcessorParams]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.params.gs_folder_bert) > 0
        # load the tokenizer, local import for parallel processing support
        from official.nlp.bert.tokenization import FullTokenizer
        self.tokenizer = FullTokenizer(
            vocab_file=os.path.join(self.params.gs_folder_bert, "vocab.txt"),
            do_lower_case=True)

        logger.info(f"Loaded Tokenizer with a vocab size of {len(self.tokenizer.vocab)}")

    def apply(self, sample: Sample) -> Sample:
        def encode_sentences(sentence1, sentence2):
            tokens1 = list(self.tokenizer.tokenize(sentence1)) + ['[SEP]']
            tokens2 = list(self.tokenizer.tokenize(sentence2)) + ['[SEP]']
            return ['[CLS]'] + tokens1 + tokens2, [0] + [0] * len(tokens1) + [1] * len(tokens2)

        word_ids, type_ids = encode_sentences(sample.inputs[Keys.InputSentence1], sample.inputs[Keys.InputSentence2])
        word_ids = self.tokenizer.convert_tokens_to_ids(word_ids)
        return sample.new_inputs(
            {
                Keys.InputWordIds: word_ids,
                Keys.InputMask: np.full(fill_value=1, shape=[len(word_ids)], dtype=np.int32),
                Keys.InputTypeIds: np.asarray(type_ids, dtype=np.int32)
            }
        )
