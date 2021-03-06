from typing import Dict
import csv
import json
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("event2mind")
class Event2MindDatasetReader(DatasetReader):
    """
    Reads instances from the Event2Mind dataset.

    This dataset is CSV and has the columns:
    Source,Event,Xintent,Xemotion,Otheremotion,Xsent,Osent

    The Xintent, Xemotion, and Otheremotion columns are JSON arrays.

    For instance:
    rocstory,PersonX talks to PersonX's mother,"[""to keep in touch""]","[""accomplished""]","[""loved""]",5.0,5.0

    Currently we only consume the event, intent and emotions, not the sentiments.

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.reader(data_file)
            # Skip header
            reader.__next__()

            for (line_num, line_parts) in enumerate(reader):
                if len(line_parts) != 7:
                    line = ','.join([str(s) for s in line_parts])
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                source_sequence = line_parts[1]
                xintents = json.loads(line_parts[2])
                xreacts = json.loads(line_parts[3])
                oreacts = json.loads(line_parts[4])
                for xintent in xintents:
                    for xreact in xreacts:
                        for oreact in oreacts:
                            yield self.text_to_instance(source_sequence, xintent, xreact, oreact)

    """
    See https://github.com/maartensap/event2mind-internal/blob/master/code/modeling/utils/preprocess.py#L80.
    """
    @staticmethod
    def _preprocess_string(tokenizer, string: str) -> str:
       word_tokens = tokenizer.tokenize(string.lower())
       words = [token.text for token in word_tokens]
       if "person y" in string.lower():
          #tokenize the string, reformat PersonY if mentioned for consistency
          ws = []
          skip = False
          for i in range(0, len(words)-1):
             # TODO(brendanr): Why not handle person x too?
             if words[i] == "person" and words[i+1] == "y":
                ws.append("persony")
                skip = True
             elif skip:
                skip = False
             else:
                ws.append(words[i])
          if not skip:
             ws.append(words[-1])
          words = ws
       # get rid of "to" or "to be" prepended to annotations
       retval = []
       first = 0
       for word in words:
          first += 1
          if word == "to" and first == 1:
             continue
          if word == "be" and first < 3:
             continue
          retval.append(word)
       if len(retval) == 0:
          retval.append("none")
       return " ".join(retval)

    def _build_target_field(self, target_string: str) -> TextField:
        processed = self._preprocess_string(self._target_tokenizer, target_string)
        tokenized_target = self._target_tokenizer.tokenize(processed)
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        return TextField(tokenized_target, self._target_token_indexers)

    @overrides
    def text_to_instance(
            self,
            source_string: str,
            xintent_string: str = None,
            xreact_string: str = None,
            oreact_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        processed = self._preprocess_string(self._source_tokenizer, source_string)
        tokenized_source = self._source_tokenizer.tokenize(processed)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if xintent_string is not None:
            if xreact_string is None:
                raise Exception("missing xreact")
            if oreact_string is None:
                raise Exception("missing oreact")
            return Instance({
                "source": source_field,
                "xintent": self._build_target_field(xintent_string),
                "xreact": self._build_target_field(xreact_string),
                "oreact": self._build_target_field(oreact_string),
                })
        else:
            return Instance({'source': source_field})
