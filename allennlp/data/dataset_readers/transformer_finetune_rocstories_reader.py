import json
import logging
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

from allennlp.data.token_indexers.openai_transformer_finetune_byte_pair_indexer import OpenaiTransformerFinetuneBytePairIndexer, CLF_TAG, CONTEXT_BEGIN_TAG, DELIMITER_TAG

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import re


@DatasetReader.register("transformer_finetune_rocstories_reader")
class TransformerFinetuneRocstoriesReader(DatasetReader):
    """
    Reads a file from the Abductive Natural Language Inference (ANLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "InputSentence1", "InputSentence2", "RandomFifthSentenceQuiz1" and
    "RandomFifthSentenceQuiz2".  We convert these keys into fields named
    "sentence_x", "sentence_z" and "sentence_y", along with a metadata field containing the
    tokenized strings of the sentence_x, sentence_y and sentence_z.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for all three sentences.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this similarly for all sentences.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 context_fields: List[str] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

        if context_fields is None:
            self.context_fields = ['InputSentence1',
                                   'InputSentence2',
                                   'InputSentence3',
                                   'InputSentence4'
                                   ]
        else:
            self.context_fields = context_fields

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as snli_file:
            logger.info("Reading ANLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = int(example["AnswerRightEnding"]) - 1

                sentence_y_1 = example["RandomFifthSentenceQuiz1"]
                sentence_y_2 = example["RandomFifthSentenceQuiz2"]
                context = " ".join([example[field] for field in self.context_fields])

                yield self.text_to_instance(context, sentence_y_1, sentence_y_2, label)

    @staticmethod
    def _text_standardize(text):
        """
        fixes some issues the spacy tokenizer had on books corpus
        also does some whitespace standardization
        """
        text = text.replace('—', '-')
        text = text.replace('–', '-')
        text = text.replace('―', '-')
        text = text.replace('…', '...')
        text = text.replace('´', "'")
        text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''',
                      r' \1 ', text)
        text = re.sub(r'\s*\n\s*', ' \n ', text)
        text = re.sub(r'[^\S\n]+', ' ', text)
        return text.strip()

    @staticmethod
    def _to_delimited_option(context, option):
        return ' '.join([
            CONTEXT_BEGIN_TAG,
            TransformerFinetuneRocstoriesReader._text_standardize(context),
            DELIMITER_TAG,
            TransformerFinetuneRocstoriesReader._text_standardize(option),
            CLF_TAG
        ])

    def text_to_instance(self,
                         context,
                         option_1,
                         option_2,
                         label) -> Instance:
        fields: Dict[str, Field] = {}
        # Tokenize strings
        context_with_option_1_str = self._to_delimited_option(context, option_1)
        context_with_option_1 = self._tokenizer.tokenize(context_with_option_1_str)
        context_with_option_2_str = self._to_delimited_option(context, option_2)
        context_with_option_2 = self._tokenizer.tokenize(context_with_option_2_str)

        option_1_field = TextField(context_with_option_1, self._token_indexers)
        option_2_field = TextField(context_with_option_2, self._token_indexers)

        fields['context_with_options'] = ListField([option_1_field, option_2_field])

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)

        metadata = {
            "context_with_option_1_str": context_with_option_1_str,
            "context_with_option_1_tokens": [x.text for x in context_with_option_1],
            "context_with_option_2_str": context_with_option_2_str,
            "context_with_option_2_tokens": [x.text for x in context_with_option_2],
        }
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
