from typing import Dict, List, Tuple

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.openai_transformer_byte_pair_indexer import OpenaiTransformerBytePairIndexer
from overrides import overrides
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary

# Special tags for finetuner

CONTEXT_BEGIN_TAG = "_start_"
DELIMITER_TAG = "_delimiter_"
CLF_TAG = "_classify_"

SPECIAL_TAGS_LIST = [CONTEXT_BEGIN_TAG, DELIMITER_TAG, CLF_TAG]


@TokenIndexer.register("openai_transformer_finetune_byte_pair")
class OpenaiTransformerFinetuneBytePairIndexer(OpenaiTransformerBytePairIndexer):
    """
    A byte pair indexer for the finetuner model. Inherited from OpenaiTransformerBytePairIndexer.
    This class can be merged into AllenNLP if this is generic enough.
    """
    special_tags: Dict[str, int] = {}

    def __init__(self,
                 encoder: Dict[str, int] = None,
                 byte_pairs: List[Tuple[str, str]] = None,
                 n_ctx: int = 512,
                 model_path: str = None) -> None:
        super().__init__(encoder, byte_pairs, n_ctx, model_path)

        for special_tag in SPECIAL_TAGS_LIST:
            idx = len(self.encoder)
            self.encoder[special_tag] = idx
            self.decoder[idx] = special_tag
            OpenaiTransformerFinetuneBytePairIndexer.special_tags[special_tag] = idx

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          _vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        text_tokens = []
        offsets = []
        offset = -1

        for token in tokens:
            # Special case for SPECIAL TAGS
            if token.text in SPECIAL_TAGS_LIST:
                bpe_tokens = [self.encoder.get(token.text)]
            else:
                bpe_tokens = [self.encoder.get(t, 0) for t in self.byte_pair_encode(token)]
            offset += len(bpe_tokens)
            offsets.append(offset)
            text_tokens.extend(bpe_tokens)

        num_tokens = len(text_tokens)

        # If there's too many tokens, that's going to cause problems.
        if num_tokens > self.n_ctx:
            raise RuntimeError(
                f"The transformer model has a maximum sequence length of {self.n_ctx} "
                f"but your byte pair encoded sequence has length {num_tokens}. "
                f"The offending text input is {tokens}.")

        # If there's too few tokens, just pad with zeros.
        text_tokens.extend(0 for _ in range(self.n_ctx - num_tokens))

        return {
            index_name: text_tokens,
            f"{index_name}-offsets": offsets,
            # add mask here according to the original tokens,
            # because calling util.get_text_field_mask on the
            # "byte pair" tokens will produce the wrong shape
            "mask": [1 for _ in offsets]
        }
