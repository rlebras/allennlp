from typing import Dict, List
import csv
import json
import logging

from overrides import overrides

from allennlp.data.fields.list_field import ListField
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import itertools

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("event2event")
class Event2EventDatasetReader(DatasetReader):
    """
    Reads instances from the Atomic dataset.

    This dataset is in CSV format and has a column 'event', as well as the columns specified in 'target_fields'

    For instance:
    event,oEffect,oReact
    PersonX uses PersonX's ___ to obtain,[],"[""annoyed"", ""angry"", ""worried""]"

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
    target_fields : ``Sequence[str]``
        List of target fields associated with an event.
    """
    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 target_fields = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._target_fields = target_fields

    @overrides
    def _read(self, file_path):

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.DictReader(data_file)
            header = reader.fieldnames

            for (line_num, line_dict) in enumerate(reader):
                if len(line_dict) != len(header):
                    line = ','.join([str(s) for s in line_dict.items()])
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))

                # Event
                source_sequence = line_dict["event"]

                # Targets
                targets = []
                for field in self._target_fields:
                    if line_dict[field] == None or line_dict[field] == "[]":
                        targets.append([""])
                    else:
                        targets.append(json.loads(line_dict[field]))

                for vals in itertools.product(*targets):
                    target_dict = {}
                    has_target = False
                    for (j, field) in self._target_fields:
                        target_dict[field] = vals[j]
                        if vals[j] != "":
                            has_target = True
                    if has_target:
                        yield self.text_to_instance(source_sequence, target_dict)

    """
    See https://github.com/maartensap/event2mind-internal/blob/master/code/modeling/utils/preprocess.py#L80.
    """
    @staticmethod
    def _preprocess_string(tokenizer, string: str) -> str:
       if str == "":
           return ""
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
   
    @staticmethod
    def _preprocess_string_better(tokenizer, string: str, append_period = False) -> str:
       string = string\
                .replace("person x","personx")\
                .replace("Person x","Personx")\
                .replace("person y","persony")\
                .replace("Person y","Persony")\
                .replace("person z","personz")\
                .replace("Person z","Personz")
       
       string = string[:-1] if string[-1] in ".,;:=!/&+\\" else string
       
       word_tokens = tokenizer.tokenize(string)
       words = [token.text for token in word_tokens]
       
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
       else:
           if append_period == True:
              retval.append(".")
       return " ".join(retval)

    def _build_target_field(self, target_string: str) -> TextField:
        if target_string == "":
            ret = TextField([Token(START_SYMBOL), Token(END_SYMBOL)], self._target_token_indexers)
        else:
            processed = self._preprocess_string_better(self._target_tokenizer, target_string)
            tokenized_target = self._target_tokenizer.tokenize(processed)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            ret = TextField(tokenized_target, self._target_token_indexers)
        return ret
    @overrides
    def text_to_instance(
            self,
            source_string: str,
            target_dict = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        processed = self._preprocess_string_better(self._source_tokenizer, source_string, append_period=True)
        tokenized_source = self._source_tokenizer.tokenize(processed)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_dict is not None:
            dict = {"source": source_field}
            for key, val in target_dict.items():
                dict[key] = self._build_target_field(val)
            return Instance(dict)
        else:
            return Instance({'source': source_field})

@DatasetReader.register("event2event_targetDomains")
class Event2EventDatasetReader(DatasetReader):
    """
    Reads instances from the Atomic dataset.

    This dataset is in CSV format and has a column 'event', as well as the columns specified in 'target_fields'

    For instance:
    event,oEffect,oReact
    PersonX uses PersonX's ___ to obtain,[],"[""annoyed"", ""angry"", ""worried""]"

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
    target_fields : ``Sequence[str]``
        List of target fields associated with an event.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 target_fields = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._target_fields = target_fields

    @overrides
    def _read(self, file_path):

        event2targetdomain = {}
        with open(cached_path(file_path), "r") as data_file:
            logger.info("(1st pass) Reading instances from lines in file at: %s", file_path)
            reader = csv.DictReader(data_file)

            for (line_num, line_dict) in enumerate(reader):
                source_sequence = line_dict["event"]
                if source_sequence not in event2targetdomain.keys():
                    event2targetdomain[source_sequence] = {}

                for field in self._target_fields:
                    if field not in event2targetdomain[source_sequence]:
                        event2targetdomain[source_sequence][field] = []

                    if not(line_dict[field] == None or line_dict[field] == "[]"):
                        annotations = json.loads(line_dict[field])
                        for a in annotations:
                            event2targetdomain[source_sequence][field].append(a)

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.DictReader(data_file)
            header = reader.fieldnames

            for (line_num, line_dict) in enumerate(reader):
                if len(line_dict) != len(header):
                    line = ','.join([str(s) for s in line_dict.items()])
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))

                # Event
                source_sequence = line_dict["event"]

                # Targets
                targets = []
                for field in self._target_fields:
                    if line_dict[field] == None or line_dict[field] == "[]":
                        targets.append([""])
                    else:
                        targets.append(json.loads(line_dict[field]))

                for vals in itertools.product(*targets):
                    target_dict = {}
                    has_target = False
                    for (j, field) in enumerate(self._target_fields):
                        target_dict[field] = vals[j]
                        field_dom = field + "_dom"
                        target_dict[field_dom] = event2targetdomain[source_sequence][field]
                        if vals[j] != "":
                            has_target = True
                    if has_target:
                        yield self.text_to_instance(source_sequence, target_dict)

    """
    See https://github.com/maartensap/event2mind-internal/blob/master/code/modeling/utils/preprocess.py#L80.
    """

    @staticmethod
    def _preprocess_string(tokenizer, string: str) -> str:
        if str == "":
            return ""
        word_tokens = tokenizer.tokenize(string.lower())
        words = [token.text for token in word_tokens]
        if "person y" in string.lower():
            # tokenize the string, reformat PersonY if mentioned for consistency
            ws = []
            skip = False
            for i in range(0, len(words) - 1):
                # TODO(brendanr): Why not handle person x too?
                if words[i] == "person" and words[i + 1] == "y":
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

    @staticmethod
    def _preprocess_string_betterNew(tokenizer, string: str, append_period=False) -> str:
        string = string \
            .replace("person x", "personx") \
            .replace("Person x", "Personx") \
            .replace("person y", "persony") \
            .replace("Person y", "Persony") \
            .replace("person z", "personz") \
            .replace("Person z", "Personz")

        string = string[:-1] if string[-1] in ".,;:=!/&+\\" else string

        word_tokens = tokenizer.tokenize(string)
        words = [token.text for token in word_tokens]

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
        else:
            if append_period == True:
                retval.append(".")
        return " ".join(retval)

    @staticmethod
    def _preprocess_string_better(tokenizer, string: str) -> str:
        string = string.lower() \
            .replace("person x", "personx") \
            .replace("person y", "persony") \
            .replace("person z", "personz")

        string = string[:-1] if string[-1] in ".,;:=!/&+\\" else string

        word_tokens = tokenizer.tokenize(string)
        words = [token.text for token in word_tokens]

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
        if target_string == "":
            ret = TextField([Token(START_SYMBOL), Token(END_SYMBOL)], self._target_token_indexers)
        else:
            processed = self._preprocess_string_better(self._target_tokenizer, target_string)
            tokenized_target = self._target_tokenizer.tokenize(processed)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            ret = TextField(tokenized_target, self._target_token_indexers)
        return ret

    def _build_target_field_dom(self, target_string_dom) -> List[TextField]:
        ret: List[TextField] = []
        for target_string in target_string_dom:
            ret.append(self._build_target_field(target_string))
        return ret

    @overrides
    def text_to_instance(
            self,
            source_string: str,
            target_dict=None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        processed = self._preprocess_string_better(self._source_tokenizer, source_string)
        tokenized_source = self._source_tokenizer.tokenize(processed)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        if target_dict is not None:
            dict = {"source": source_field}
            for key in self._target_fields:
                dict[key] = self._build_target_field(target_dict[key])
                key_dom = key + "_dom"
                dom = self._build_target_field_dom(target_dict[key_dom])
                if (len(dom) > 0):
                    dict[key_dom] = ListField(dom)
                else:
                    # in my actual code dummyseq is a populated SequenceLabelField
                    dict[key_dom] = ListField([source_field.empty_field()])
            return Instance(dict)
        else:
            return Instance({'source': source_field})
