from typing import Optional, Dict, List, Any

import torch
from allennlp.common.from_params import FromParams
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.openai_transformer import OpenaiTransformer
from allennlp.nn import RegularizerApplicator
from allennlp.nn.util import get_range_vector, get_device_of
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from allennlp.data.token_indexers.openai_transformer_finetune_byte_pair_indexer import OpenaiTransformerFinetuneBytePairIndexer, CLF_TAG


class LMHead(torch.nn.Module):
    def __init__(self, transformer_model: OpenaiTransformer):
        super(LMHead, self).__init__()
        self.embedding_dim = transformer_model.embed.embedding_dim
        embed_shape = transformer_model.embed.weight.shape
        self.decoder = torch.nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = transformer_model.embed.weight # Tied weights

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.embedding_dim)
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class MultipleChoiceTaskHead(torch.nn.Module):
    def __init__(self,
                 transformer_model: OpenaiTransformer,
                 clf_token_index,
                 num_options,
                 dropout=0.1):
        super(MultipleChoiceTaskHead, self).__init__()
        self.embedding_dim = transformer_model.embed.embedding_dim
        self.dropout = torch.nn.Dropout2d(dropout)
        self.linear = torch.nn.Linear(self.embedding_dim, 1)
        self.clf_token_index = clf_token_index
        self.num_options = num_options

        torch.nn.init.normal_(self.linear.weight, std = 0.02)
        torch.nn.init.normal_(self.linear.bias, 0)

    def forward(self, h: torch.Tensor, x: torch.Tensor, x_mask: torch.Tensor):

        # Classification logits. Pick the hidden state at the special _classify_ token
        ys = x_mask.sum(1)
        xs = torch.LongTensor(range(ys.size(0)))

        clf_h = torch.stack([h[x, y-1, :] for x, y in zip(xs, ys)])

        # Change to batch x no_options x dim x 1 tensor
        clf_h = clf_h.view(-1, x.size(0), self.embedding_dim, 1)

        clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)

        clf_h = clf_h.contiguous().view(-1, self.embedding_dim)
        clf_logits = self.linear(clf_h)

        return clf_logits.view(-1, self.num_options)


class MultipleChoiceLoss:
    def __init__(self,
                 lm_loss=torch.nn.CrossEntropyLoss(reduce=False),
                 clf_loss=torch.nn.CrossEntropyLoss(reduce=False),
                 lm_loss_weight=0.5):
        self.lm_loss = lm_loss
        self.clf_loss = clf_loss
        self.lm_loss_weight = lm_loss_weight

    def __call__(self,
                 stacked_context_with_options,
                 gold_label,
                 clf_logits,
                 lm_logits=None):

        _x = stacked_context_with_options['openai_transformer']
        offsets = stacked_context_with_options['openai_transformer-offsets']
        mask = stacked_context_with_options['mask']

        losses_dict = {}
        if lm_logits is not None:
            # x_shifted = X[:, :, 1:, 0].contiguous().view(-1)
            # Get X based on the offsets

            range_vector = get_range_vector(_x.size(0), device=get_device_of(_x)).unsqueeze(1)

            _x_selected = _x[range_vector, offsets]
            x_shifted = _x_selected[:, 1:].contiguous().view(-1)

            lm_losses = self.lm_loss(lm_logits, x_shifted)
            lm_losses = lm_losses.view(-1, _x_selected.size(1) - 1)
            lm_losses = lm_losses * mask[:, 1:].float()
            lm_losses = lm_losses.sum(1) / torch.sum(mask[:, 1:], 1).float()
            losses_dict['lm_loss'] = lm_losses.sum() * self.lm_loss_weight

        # Classification loss
        clf_losses = self.clf_loss(clf_logits, gold_label)
        losses_dict['clf_loss'] = clf_losses.sum()

        return losses_dict


@Model.register("transformer_finetuner")
class OpenaiTransformerFinetuner(Model, FromParams):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 task_head_type: str,
                 num_options: int,
                 include_lm_loss: bool,
                 lm_loss_weight: float = 0.5,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._transformer_model = \
            self._text_field_embedder._token_embedders['openai_transformer']._transformer
        self.lm_head = LMHead(self._transformer_model)
        self.clf_token_index = OpenaiTransformerFinetuneBytePairIndexer.special_tags[CLF_TAG]
        self.include_lm_loss = include_lm_loss

        # Implement multiple choice finetuner
        if task_head_type == "mc":
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce=False)
            self.task_head = MultipleChoiceTaskHead(self._transformer_model,
                                                    clf_token_index=self.clf_token_index,
                                                    num_options=num_options)
            self.loss_fn = \
                MultipleChoiceLoss(lm_loss=cross_entropy_loss,
                                   clf_loss=cross_entropy_loss,
                                   lm_loss_weight=lm_loss_weight)
        else:
            raise NotImplementedError

        self.metrics = {"accuracy": CategoricalAccuracy()}

    def forward(self,
                context_with_options,
                label,
                metadata: List[Dict[str, Any]]):

        stacked_context_with_options = {}
        for k, v in context_with_options.items():
            batch_times_options = context_with_options[k].size(0) * context_with_options[k].size(1)
            stacked_context_with_options[k] = v.view(batch_times_options, -1)

        context_option_embeddings = self._text_field_embedder(stacked_context_with_options)

        output_dict = {}
        if self.include_lm_loss:
            lm_logits = self.lm_head(context_option_embeddings)
            output_dict['lm_logits'] = lm_logits
        else:
            lm_logits = None

        clf_logits = self.task_head(context_option_embeddings, stacked_context_with_options[
            'openai_transformer'], stacked_context_with_options['mask'])

        output_dict["clf_logits"] = clf_logits

        if label is not None:
            losses = self.loss_fn(stacked_context_with_options, label, clf_logits, lm_logits)
            output_dict['loss'] = losses['clf_loss'] + losses['lm_loss']

            for metric in self.metrics.values():
                metric(clf_logits, label)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
