from typing import Dict
import csv
import json
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import numpy as np
import itertools

from IPython import embed as ip_embed

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("event2event")
class Event2EventDatasetReader(DatasetReader):
    """
    Reads instances from the Event2Event dataset.

    This dataset is CSV and has the columns:
    ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]

    For instance:
    ...

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
                 lazy: bool = False,
                 target_fields = None) -> None:
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

                # source_seq : event
                source_sequence = line_dict["event"]

                # HACK to make elmo better potentially
                source_sequence += " ."

                # ["oEffect", "oReact", "oWant", "xAttr" , "xEffect", "xIntent", "xNeed", "xReact", "xWant"]
                # ip_embed()
                target_indices = range(0, len(self._target_fields))
                targets = []
                for field in self._target_fields:
                    if line_dict[field] == None or line_dict[field] == "[]":
                        targets.append([""])
                    else:
                        targets.append(json.loads(line_dict[field]))

                for vals in itertools.product(*targets):
                    target_dict = {}
                    has_target = False
                    for j in target_indices:
                        target_dict[self._target_fields[j]] = vals[j]
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


@DatasetReader.register("atomic")
class AtomicDatasetReader(DatasetReader):
    """
    Reads instances from the Event2Event dataset.

    This dataset is CSV and has the columns:
    ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]

    For instance:
    ...

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
    dummy_instances_for_vocab_generation : ``bool`` (optional, default=False)
        Whether to generate instances that use each token of input precisely
        once. Normally we instead generate all combinations of Source and target columns
        which distort the underlying token counts. This flag should be used exclusively
        with the ``dry-run`` command as the instances generated will be nonsensical
        outside the context of vocabulary generation.
    """

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 dummy_instances_for_vocab_generation: bool = False,
                 lazy: bool = False,
                 target_fields=None) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._target_fields = target_fields
        self._dummy_instances_for_vocab_generation = dummy_instances_for_vocab_generation

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

                # source_seq : event
                source_sequence = line_dict["event"]

                # HACK to make elmo better potentially
                source_sequence += " ."

                # ["oEffect", "oReact", "oWant", "xAttr" , "xEffect", "xIntent", "xNeed", "xReact", "xWant"]
                target_indices = range(0, len(self._target_fields))
                targets = []
                for field in self._target_fields:
                    if line_dict[field] == None or line_dict[field] == "[]":
                        targets.append([""])
                    else:
                        targets.append(json.loads(line_dict[field]))

                # Generate all combinations.
                if not self._dummy_instances_for_vocab_generation:
                    for vals in itertools.product(*targets):
                        target_dict = {}
                        has_target = False
                        for j in target_indices:
                            target_dict[self._target_fields[j]] = vals[j]
                            if vals[j] != "":
                                has_target = True
                        if has_target:
                            yield self.text_to_instance(source_sequence, target_dict)
                # Generate instances where each token of input appears once.
                else:
                    target_dict = {}
                    for j in target_indices:
                        target_dict[self._target_fields[j]] = "none"
                    yield self.text_to_instance(source_sequence, target_dict)

                    for j in target_indices:
                        target_field = self._target_fields[j]
                        for target_value in targets[j]:
                            target_dict = {}
                            for j in target_indices:
                                target_dict[self._target_fields[j]] = "none"
                            target_dict[target_field] = target_value
                            yield self.text_to_instance("none", target_dict)


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
    def _preprocess_string_better(tokenizer, string: str, append_period=False) -> str:
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
            target_dict=None) -> Instance:  # type: ignore
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


@DatasetReader.register("event2event_wDimGroups")
class Event2Event_wDimGroupsDatasetReader(Event2EventDatasetReader):
    """
    Reads instances from the Event2Event dataset.
    This will yield instances with:
     - one event (e.g., "PersonX moved apartments")
     - one inference type (e.g., "xIntent")
     - one inference phrase (e.g. "to get a change of scenery")

    This dataset is CSV and has the columns:
    ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]

    For instance:
    ...

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
                 # dim_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 # dim_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

        self._dim_map = {d:i for i,d  in enumerate(
            ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"])}
        
    @overrides
    def _read(self, file_path):

        target_fields = ["oEffect", "oReact", "oWant", "xAttr", "xEffect", "xIntent", "xNeed", "xReact", "xWant"]

        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            reader = csv.DictReader(data_file)
            target_fields = reader.fieldnames
            
            for (line_num, line_dict) in enumerate(reader):
                if len(line_dict) != len(target_fields):
                    line = ','.join([str(s) for s in line_dict.items()])
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                
                # source_seq : event
                source_sequence = line_dict["event"]
                del line_dict["event"]
                # ["oEffect", "oReact", "oWant", "xAttr" , "xEffect", "xIntent", "xNeed", "xReact", "xWant"]

                for dim, seqs in line_dict.items():
                    if not seqs or seqs == '[]':
                        continue
                    for seq in json.loads(seqs):
                        target_dict = {
                            "target_seq": seq,
                            "dim": dim
                        }
                        yield self.text_to_instance(source_sequence, target_dict)
                
                # target_indices = range(0, len(target_fields))
                # targets_dict = {}
                # targets = []
                # for i in target_indices:
                #     field = target_fields[i]
                #     if line_parts[i + 1] == None or line_parts[i + 1] == "[]":
                #         targets_dict[field] = [""]
                #         targets.append([""])
                #     else:
                #         targets_dict[field] = json.loads(line_parts[i + 1])
                #         targets.append(json.loads(line_parts[i + 1]))

                # for vals in itertools.product(*targets):
                #     target_dict = {}
                #     for j in target_indices:
                #         target_dict[target_fields[j]] = vals[j]
                #     yield self.text_to_instance(source_sequence, target_dict)

    def _build_target_field(self, target_string: str) -> TextField:
        processed = self._preprocess_string(self._target_tokenizer, target_string)
        tokenized_target = self._target_tokenizer.tokenize(processed)
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        return TextField(tokenized_target, self._target_token_indexers)

    def _build_target_dim(self, dim: str):
        ix = self._dim_map[dim]
        arr = np.array([0]*len(self._dim_map))
        arr[ix] = 1
        return ArrayField(arr)
        
    @overrides
    def text_to_instance(
            self,
            source_string: str,
            target_dict = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        processed = self._preprocess_string(self._source_tokenizer, source_string)
        tokenized_source = self._source_tokenizer.tokenize(processed)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
        
        if target_dict is not None:
            t_d = {"source": source_field}
            t_d["target_seq"] = self._build_target_field(target_dict["target_seq"])
            t_d["dim"] = self._build_target_dim(target_dict["dim"])

            return Instance(t_d)
        else:
            return Instance({'source': source_field})
