from typing import Dict, Tuple

import numpy
from overrides import overrides

from IPython import embed as ip_embed

import torch
from torch.nn.modules.rnn import GRUCell, LSTMCell
from torch.nn.modules.linear import Linear
from torch import nn
import torch.nn.functional as F
from allennlp.training.metrics.metric import Metric

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from allennlp.training.metrics import UnigramRecall, RougeL, RougeN, BleuN, BatchedAverage

from collections import Counter, OrderedDict

class StateDecoder:
    def __init__(self, name, event2event, num_classes, input_dim, output_dim, metrics):
        self._embedder = Embedding(num_classes, input_dim)
        event2event.add_module("{}_embedder".format(name), self._embedder)
        self._decoder_cell = GRUCell(input_dim, output_dim)
        event2event.add_module("{}_decoder_cell".format(name), self._decoder_cell)
        self._output_dim = output_dim
        self._output_projection_layer = Linear(output_dim, num_classes)
        event2event.add_module("{}_output_project_layer".format(name), self._output_projection_layer)
        self._recalls = {}
        #self._xent = BatchedAverage()
        #for m in metrics:
        #    # WARNING!!!
        #    if m == "unigram_recall":
        #        self._recalls[m] = Metric.by_name(m)()

        self._bleu = BleuN(n=2)
        #self._bleu1 = BleuN(n=1)
        #self._bleu4 = BleuN(n=4)
        #self._rouge = RougeL()
        #self._rouge0 = RougeL(alpha=0.)
        #self._rouge1 = RougeL(alpha=1.)

    def _transform_init_state(self,init_hs):
        """Does nothing"""
        return init_hs
    
    def greedy_search(self, final_encoder_output, target_tokens, # target_embedder,
                      # decoder_cell, output_projection_layer,
                      early_fusion=None,
                      batch_average_loss=True):

        target_embedder = self._embedder
        decoder_cell = self._decoder_cell
        output_projection_layer = self._output_projection_layer

        final_encoder_output = self._transform_init_state(final_encoder_output)
        
        targets = target_tokens["tokens"]
        target_sequence_length = targets.size()[1]
        # The last input from the target is either padding or the end symbol. Either way, we
        # don't have to process it.
        # TODO(brendanr): Something about this is suspicious. As in will we
        # maybe have difficulty learning to output the end symbol? Maybe
        # it's fine since this will make num_decoding_steps the length of
        # the longest sequence and most targets will be shorter? Still...
        num_decoding_steps = target_sequence_length - 1

        decoder_hidden = final_encoder_output
        step_logits = []
        for timestep in range(num_decoding_steps):
            # See https://github.com/allenai/allennlp/issues/1134.
            # TODO(brendanr): Grok this.
            input_choices = targets[:, timestep]
            decoder_input = target_embedder(input_choices)
            
            if not early_fusion is None:
                decoder_input = torch.cat([decoder_input,early_fusion],dim=1)

            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size, num_classes)
            output_projections = output_projection_layer(decoder_hidden)
            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))
        # (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        target_mask = get_text_field_mask(target_tokens)

        empty_target_mask = []
        for i, t in enumerate(targets.data):
            if numpy.count_nonzero(t.data) == 2:
                empty_target_mask.append([0]*len(target_mask.data[i]))
            else:
                empty_target_mask.append([1]*len(target_mask.data[i]))
        empty_target_mask_tsr = torch.tensor(empty_target_mask, device=target_mask.device)

        target_mask = target_mask*empty_target_mask_tsr
        loss = Event2Event._get_loss(logits, targets, target_mask, batch_average=batch_average_loss)
        count = (target_mask.sum(1) > 0).sum()
        return loss, count

    
    def beam_search(self,
                    final_encoder_output: torch.LongTensor,
                    k: int,
                    num_decoding_steps: int,
                    batch_size: int,
                    source_mask,
                    start_ix,
                    end_ix,
                    num_classes,
                    early_fusion=None) -> Tuple[torch.Tensor, torch.Tensor]:

        final_encoder_output = self._transform_init_state(final_encoder_output)
        
        target_embedder = self._embedder
        decoder_cell = self._decoder_cell
        output_projection_layer = self._output_projection_layer

        # List of (batchsize, k) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions = []
        # TODO(brendanr): Fix this comment.
        # List of (batchsize, k) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from. This is aligned with
        # predictions so that backpointer[t][i][n] corresponds to
        # predictions[t][n].
        backpointers = []
        # List of (batchsize * k,) tensors.
        # TODO(brendanr): Just keep last
        log_probabilities = []

        # Timestep 1
        start_predictions = source_mask.new_full((batch_size,), fill_value=start_ix)
        start_decoder_input = target_embedder(start_predictions)
        if not early_fusion is None:
            start_decoder_input = torch.cat([start_decoder_input,early_fusion],dim=1)
        start_decoder_hidden = decoder_cell(start_decoder_input, final_encoder_output)
        start_output_projections = output_projection_layer(start_decoder_hidden)
        start_class_log_probabilities = F.log_softmax(start_output_projections, dim=-1)
        start_top_log_probabilities, start_predicted_classes = start_class_log_probabilities.topk(k)

        # Set starting values
        # [(batch_size, k)]
        log_probabilities.append(start_top_log_probabilities)
        # [(batch_size, k)]
        predictions.append(start_predicted_classes)
        # Set the same hidden state for each element in beam.
        # (batch_size * k, _decoder_output_dim)
        decoder_hidden = start_decoder_hidden.\
            unsqueeze(1).expand(batch_size, k, self._output_dim).\
            reshape(batch_size * k, self._output_dim)

        # Log probability tensor that mandates that the end token is selected.
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * k, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, end_ix] = 0.0

        for timestep in range(num_decoding_steps - 1):
            # (batch_size * k,)
            last_predictions = predictions[-1].reshape(batch_size * k)
            decoder_input = target_embedder(last_predictions)
            
            if not early_fusion is None:
                decoder_input = torch.cat([decoder_input,torch.cat([early_fusion]*k)],dim=1)
            
            # reshape(batch_size * k, self._output_dim)
            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size * k, num_classes)
            output_projections = output_projection_layer(decoder_hidden)

            # (batch_size * k, num_classes)
            class_log_probabilities = F.log_softmax(output_projections, dim=-1)

            # (batch_size * k, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(batch_size * k, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == end_ix,
                log_probs_after_end,
                class_log_probabilities
            )

            # (batch_size * k, k), (batch_size * k, k)
            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(k)
            # TODO(brendanr): Normalize for length?
            # (batch_size * k, k)
            expanded_last_log_probabilities = log_probabilities[-1].\
                unsqueeze(2).\
                expand(batch_size, k, k).\
                reshape(batch_size * k, k)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            reshaped_top_log_probabilities = summed_top_log_probabilities.reshape(batch_size, k * k)
            reshaped_predicted_classes = predicted_classes.reshape(batch_size, k * k)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_top_log_probabilities.topk(k)
            # TODO(brendanr): Something about this is weird. Why do restricted_predicted_classes == restricted_beam_indices?
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)

            log_probabilities.append(restricted_beam_log_probs)
            predictions.append(restricted_predicted_classes)
            backpointer = restricted_beam_indices / k
            backpointers.append(backpointer)
            expanded_backpointer = backpointer.unsqueeze(2).expand(batch_size, k, self._output_dim)
            decoder_hidden = decoder_hidden.\
                    reshape(batch_size, k, self._output_dim).\
                    gather(1, expanded_backpointer).\
                    reshape(batch_size * k, self._output_dim)

        if len(predictions) != num_decoding_steps:
            raise RuntimeError("len(predictions) not equal to num_decoding_steps")

        if len(backpointers) != num_decoding_steps - 1:
            raise RuntimeError("len(backpointers) not equal to num_decoding_steps")

        # Reconstruct the sequences.
        reconstructed_predictions = [predictions[num_decoding_steps - 1].unsqueeze(2)]
        if num_decoding_steps > 1:
            cur_backpointers = backpointers[num_decoding_steps - 2]
            for timestep in range(num_decoding_steps - 2, 0, -1):
                cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
                reconstructed_predictions.append(cur_preds)
                cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)
            final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(final_preds)
            # We don't add the start tokens here. They are implicit.

        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        return (all_predictions, log_probabilities[-1])

            
class StateDecoderLinear(StateDecoder):
    def __init__(self, name, event2event, num_classes, input_dim,
                 init_dim, output_dim, metrics, slices=None):
        self._name = name
        self._embedder = Embedding(num_classes, input_dim)
        event2event.add_module("{}_embedder".format(name), self._embedder)

        self._init_projection = Linear(init_dim, output_dim)
        event2event.add_module("{}_init_project".format(name), self._init_projection)
        
        self._decoder_cell = GRUCell(input_dim, output_dim)
        event2event.add_module("{}_decoder_cell".format(name), self._decoder_cell)
        self._output_dim = output_dim
        self._output_projection_layer = Linear(output_dim, num_classes)
        event2event.add_module("{}_output_project_layer".format(name), self._output_projection_layer)
        self._recalls = {}
        self._xent = BatchedAverage()
        for m in metrics:
            self._recalls[m] = Metric.by_name(m)()

        self._bleu = BatchedAverage()
        self._rouge = BatchedAverage()
        self.slices = slices
    
    def _transform_init_state(self,init_hs):
        if self.slices:
            init_hs_ = init_hs[:,self.slices[0]:self.slices[1]]
        else:
            init_hs_ = init_hs
        init_hs__ = self._init_projection(init_hs_)
        return init_hs__

        
        
class StateDecoderEarlyFusion:
    """ Input dim of RNN is word_emb_dim + numgroups"""
    def __init__(self, name, event2event, num_classes, input_dim, ef_dim, output_dim):
        self._embedder = Embedding(num_classes, input_dim)
        event2event.add_module("{}_embedder".format(name), self._embedder)
        self._decoder_cell = GRUCell(input_dim + ef_dim, output_dim)
        event2event.add_module("{}_decoder_cell".format(name), self._decoder_cell)
        self._output_dim = output_dim
        self._output_projection_layer = Linear(output_dim, num_classes)
        event2event.add_module("{}_output_project_layer".format(name), self._output_projection_layer)
        
        self._recalls = {}
        for n in event2event.dim_names:
            # self._recalls[n] = UnigramRecall()
            # self._recalls[n] = BleuN(n=2)
            self._recalls[n] = RougeL()


@Model.register("event2event")
class Event2Event(Model):
    """
    This ``Event2Event`` class is a :class:`Model` which takes an event
    sequence, encodes it, and then uses the encoded representation to decode
    several mental state sequences.

    See: https://www.semanticscholar.org/paper/Event2Mind/b89f8a9b2192a8f2018eead6b135ed30a1f2144d

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2VecEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 decoder_hidden_size: int = None,
                 target_embedding_dim: int = None,
                 target_fields = None,
                 metrics = None) -> None:
        super(Event2Event, self).__init__(vocab)
        # TODO(brendanr): Hack the embeddings here like initWEmb in modeling/utils/preprocess.py?
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._embedding_dropout = nn.Dropout(0.2)
        self._target_fields = target_fields

        print("vocab: ", vocab)
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        print("num classes: ", self.vocab.get_vocab_size(self._target_namespace))
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = decoder_hidden_size or self._encoder_output_dim
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()

        self._states: Dict[str, Event2Event.StateDecoder] = {}
        for field in self._target_fields:
            self._states[field] = StateDecoder(
                 name=field, event2event=self, num_classes=num_classes,
                 input_dim=target_embedding_dim, output_dim=self._decoder_output_dim,
                 metrics=metrics
            )
            #self._states[field] = StateDecoderLinear(
            #    name=field, event2event=self, num_classes=num_classes,
            #    input_dim=target_embedding_dim, init_dim=self._encoder_output_dim,
            #    output_dim=self._decoder_output_dim,
            #    metrics=metrics
            #)
    
    def _update_recall(self, all_top_k_predictions, target_tokens, target_recall):
        targets = target_tokens["tokens"]
        target_mask = get_text_field_mask(target_tokens)

        empty_target_mask = []
        for i, t in enumerate(targets.data):
            if numpy.count_nonzero(t.data) == 2:
                empty_target_mask.append([0]*len(target_mask.data[i]))
            else:
                empty_target_mask.append([1]*len(target_mask.data[i]))
        empty_target_mask_tsr = torch.tensor(empty_target_mask, device=target_mask.device)

        target_mask = target_mask * empty_target_mask_tsr

        # See comment in _get_loss.
        # TODO(brendanr): Do we need contiguous here?
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        target_recall(
                all_top_k_predictions,
                relevant_targets,
                relevant_mask,
                self._end_index
        )

    def _update_recalls(self, all_top_k_predictions, target_tokens, target_recalls):
        for metric_name, metric in target_recalls.items():
            self._update_recall(all_top_k_predictions, target_tokens, metric)

    def _get_num_decoding_steps(self, target_tokens):
        if target_tokens:
            targets = target_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            return target_sequence_length - 1
        else:
            return self._max_decoding_steps


    @overrides
    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                **target_tokens) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the target sequences.

        Parameters
        ----------
        source : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens :
           Dictionary from name to output of ``Textfield.as_array()`` applied on target
           ``TextField``. We assume that the target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        # TODO(brendanr): Revisit dropout.
        embedded_input = self._embedding_dropout(self._source_embedder(source))
        batch_size, _, _ = embedded_input.size()
        source_mask = get_text_field_mask(source)

        print("num classes: ", self.vocab.get_vocab_size(self._target_namespace))

        # (batch_size, encoder_output_dim)
        final_encoder_output = self._encoder(embedded_input, source_mask)
        output_dict = {}

        # Perform greedy search so we can get the loss.
        if target_tokens:
            #if target_tokens.keys() != self._states.keys():
            #    target_only = target_tokens.keys() - self._states.keys()
            #    states_only = self._states.keys() - target_tokens.keys()
            #    raise Exception("Mismatch between target_tokens and self._states. Keys in " +
            #            "targets only: {} Keys in states only: {}".format(target_only, states_only))
            total_loss = 0
            loss_count = 0

            for name, state in self._states.items():

                # loss, count = self.greedy_search(
                #     final_encoder_output,
                #     target_tokens[name],
                #     state._embedder,
                #     state._decoder_cell,
                #     state._output_projection_layer
                # )
                loss, count = state.greedy_search(
                    final_encoder_output,target_tokens[name])
                
                # total xent over non-zero targets
                output_dict["{}_loss".format(name)] = loss * count.float()
                output_dict["{}_count".format(name)] = count
                
                total_loss += loss
                loss_count = loss_count + 1

            # Average loss for interpretability.
            if loss_count == 0:
                output_dict["loss"] = 1.0
            else:
                output_dict["loss"] = total_loss / loss_count

        # Perform beam search to obtain the predictions.
        if not self.training:
            for name, state in self._states.items():
                # (batch_size, k, num_decoding_steps)
                
                # (all_top_k_predictions, log_probabilities) = self.beam_search(
                #         final_encoder_output_,
                #         10,
                #         self._get_num_decoding_steps(target_tokens.get(name)),
                #         batch_size,
                #         source_mask,
                #         state._embedder,
                #         state._decoder_cell,
                #         state._output_projection_layer
                # )
                (all_top_k_predictions, log_probabilities) = state.beam_search(
                    final_encoder_output,
                    10,
                    self._max_decoding_steps,
                    batch_size,
                    source_mask,
                    self._start_index,
                    self._end_index,
                    self.vocab.get_vocab_size(self._target_namespace))
                
                if target_tokens:
                    self._update_recalls(all_top_k_predictions, target_tokens[name], state._recalls)
                    # also update loss counter
                    #state._xent(output_dict[name+"_loss"],output_dict[name+"_count"])
                    refs = target_tokens[name + "_dom"]["tokens"][:, :, 1:].contiguous()
                    for pred in all_top_k_predictions:
                        state._bleu(refs, pred, mask = None, end_index = self._end_index, dont_count_empty_predictions = True)
                        #state._bleu1(refs, pred, mask = None, end_index = self._end_index, dont_count_empty_predictions = True)
                        #state._bleu4(refs, pred, mask = None, end_index = self._end_index, dont_count_empty_predictions = True)
                        #state._rouge(refs, pred, mask = None, end_index = self._end_index, dont_count_empty_predictions = True)
                        #state._rouge0(refs, pred, mask=None, end_index=self._end_index, dont_count_empty_predictions=True)
                        #state._rouge1(refs, pred, mask=None, end_index=self._end_index, dont_count_empty_predictions=True)

                output_dict["{}_top_k_predictions".format(name)] = all_top_k_predictions
                output_dict["{}_top_k_log_probabilities".format(name)] = log_probabilities

        return output_dict

    # Returns the loss.
    def greedy_search(self, final_encoder_output, target_tokens, target_embedder,
                      decoder_cell, output_projection_layer, early_fusion=None,
                      batch_average_loss=True):
        targets = target_tokens["tokens"]
        target_sequence_length = targets.size()[1]
        # The last input from the target is either padding or the end symbol. Either way, we
        # don't have to process it.
        # TODO(brendanr): Something about this is suspicious. As in will we
        # maybe have difficulty learning to output the end symbol? Maybe
        # it's fine since this will make num_decoding_steps the length of
        # the longest sequence and most targets will be shorter? Still...
        num_decoding_steps = target_sequence_length - 1

        decoder_hidden = final_encoder_output
        step_logits = []
        for timestep in range(num_decoding_steps):
            # See https://github.com/allenai/allennlp/issues/1134.
            # TODO(brendanr): Grok this.
            input_choices = targets[:, timestep]
            decoder_input = target_embedder(input_choices)
            
            if not early_fusion is None:
                decoder_input = torch.cat([decoder_input,early_fusion],dim=1)

            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size, num_classes)
            output_projections = output_projection_layer(decoder_hidden)
            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))
        # (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        target_mask = get_text_field_mask(target_tokens)

        empty_target_mask = []
        for i, t in enumerate(targets.data):
            if numpy.count_nonzero(t.data) == 2:
                empty_target_mask.append([0]*len(target_mask.data[i]))
            else:
                empty_target_mask.append([1]*len(target_mask.data[i]))
        empty_target_mask_tsr = torch.tensor(empty_target_mask, device=target_mask.device)

        target_mask = target_mask*empty_target_mask_tsr
        loss = self._get_loss(logits, targets, target_mask, batch_average=batch_average_loss)
        count = (target_mask.sum(1) > 0).sum()
        return loss, count

    def beam_search(self,
                    final_encoder_output: torch.LongTensor,
                    k: int,
                    num_decoding_steps: int,
                    batch_size: int,
                    source_mask,
                    target_embedder,
                    decoder_cell,
                    output_projection_layer,
                    early_fusion=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # List of (batchsize, k) tensors. One for each time step. Does not
        # include the start symbols, which are implicit.
        predictions = []
        # TODO(brendanr): Fix this comment.
        # List of (batchsize, k) tensors. One for each time step. None for
        # the first.  Stores the index n for the parent prediction, i.e.
        # predictions[t-1][i][n], that it came from. This is aligned with
        # predictions so that backpointer[t][i][n] corresponds to
        # predictions[t][n].
        backpointers = []
        # List of (batchsize * k,) tensors.
        # TODO(brendanr): Just keep last
        log_probabilities = []

        # Timestep 1
        start_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)
        start_decoder_input = target_embedder(start_predictions)
        if not early_fusion is None:
            start_decoder_input = torch.cat([start_decoder_input,early_fusion],dim=1)
        start_decoder_hidden = decoder_cell(start_decoder_input, final_encoder_output)
        start_output_projections = output_projection_layer(start_decoder_hidden)
        start_class_log_probabilities = F.log_softmax(start_output_projections, dim=-1)
        start_top_log_probabilities, start_predicted_classes = start_class_log_probabilities.topk(k)

        # Set starting values
        # [(batch_size, k)]
        log_probabilities.append(start_top_log_probabilities)
        # [(batch_size, k)]
        predictions.append(start_predicted_classes)
        # Set the same hidden state for each element in beam.
        # (batch_size * k, _decoder_output_dim)
        decoder_hidden = start_decoder_hidden.\
            unsqueeze(1).expand(batch_size, k, self._decoder_output_dim).\
            reshape(batch_size * k, self._decoder_output_dim)

        # Log probability tensor that mandates that the end token is selected.
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * k, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.0

        for timestep in range(num_decoding_steps - 1):
            # (batch_size * k,)
            last_predictions = predictions[-1].reshape(batch_size * k)
            decoder_input = target_embedder(last_predictions)
            
            if not early_fusion is None:
                decoder_input = torch.cat([decoder_input,torch.cat([early_fusion]*k)],dim=1)
            
            # reshape(batch_size * k, self._decoder_output_dim)
            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size * k, num_classes)
            output_projections = output_projection_layer(decoder_hidden)

            # (batch_size * k, num_classes)
            class_log_probabilities = F.log_softmax(output_projections, dim=-1)

            # (batch_size * k, num_classes)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(batch_size * k, num_classes)
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )

            # (batch_size * k, k), (batch_size * k, k)
            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(k)
            # TODO(brendanr): Normalize for length?
            # (batch_size * k, k)
            expanded_last_log_probabilities = log_probabilities[-1].\
                unsqueeze(2).\
                expand(batch_size, k, k).\
                reshape(batch_size * k, k)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities

            reshaped_top_log_probabilities = summed_top_log_probabilities.reshape(batch_size, k * k)
            reshaped_predicted_classes = predicted_classes.reshape(batch_size, k * k)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_top_log_probabilities.topk(k)
            # TODO(brendanr): Something about this is weird. Why do restricted_predicted_classes == restricted_beam_indices?
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)

            log_probabilities.append(restricted_beam_log_probs)
            predictions.append(restricted_predicted_classes)
            backpointer = restricted_beam_indices / k
            backpointers.append(backpointer)
            expanded_backpointer = backpointer.unsqueeze(2).expand(batch_size, k, self._decoder_output_dim)
            decoder_hidden = decoder_hidden.\
                    reshape(batch_size, k, self._decoder_output_dim).\
                    gather(1, expanded_backpointer).\
                    reshape(batch_size * k, self._decoder_output_dim)

        if len(predictions) != num_decoding_steps:
            raise RuntimeError("len(predictions) not equal to num_decoding_steps")

        if len(backpointers) != num_decoding_steps - 1:
            raise RuntimeError("len(backpointers) not equal to num_decoding_steps")

        # Reconstruct the sequences.
        reconstructed_predictions = [predictions[num_decoding_steps - 1].unsqueeze(2)]
        if num_decoding_steps > 1:
            cur_backpointers = backpointers[num_decoding_steps - 2]
            for timestep in range(num_decoding_steps - 2, 0, -1):
                cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
                reconstructed_predictions.append(cur_preds)
                cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)
            final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(final_preds)
            # We don't add the start tokens here. They are implicit.

        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)
        return (all_predictions, log_probabilities[-1])

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor,
                  batch_average: bool = True) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask,
            batch_average=batch_average)

        return loss

    def decode_all(self, predicted_indices: torch.Tensor):
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds fields for the tokens to the ``output_dict``.
        """
        for name, state in self._states.items():
            top_k_predicted_indices = output_dict["{}_top_k_predictions".format(name)][0]
            output_dict["{}_top_k_predicted_tokens".format(name)] = [self.decode_all(top_k_predicted_indices)]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        # Recall@10 needs beam search which doesn't happen during training.
        if not self.training:
            for name, state in self._states.items():
                for metric_name, metric in state._recalls.items():
                    all_metrics[name+"_" + metric_name] = metric.get_metric(reset=reset)
                # also adding cross-entropy
                #all_metrics[name+"_loss"] = state._xent.get_metric(reset=reset)
                all_metrics[name+"_bleu"] = state._bleu.get_metric(reset=reset)
                #all_metrics[name+"_bleu1"] = state._bleu1.get_metric(reset=reset)
                #all_metrics[name+"_bleu4"] = state._bleu4.get_metric(reset=reset)
                #all_metrics[name+"_rouge"] = state._rouge.get_metric(reset=reset)
                #all_metrics[name+"_rouge0"] = state._rouge0.get_metric(reset=reset)
                #all_metrics[name+"_rouge1"] = state._rouge1.get_metric(reset=reset)

        return all_metrics


@Model.register("event2event_wDimGroups")
class Event2Event_wDimGroups(Event2Event):
    """
    This ``Event2Event`` class is a :class:`Model` which takes an event
    sequence, encodes it, and then uses the encoded representation to decode
    several mental state sequences.

    See: https://www.semanticscholar.org/paper/Event2Mind/b89f8a9b2192a8f2018eead6b135ed30a1f2144d

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2VecEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 num_dim_groups: int,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None) -> None:
        super(Event2Event, self).__init__(vocab)
        # TODO(brendanr): Hack the embeddings here like initWEmb in modeling/utils/preprocess.py?
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._embedding_dropout = nn.Dropout(0.2)
        
        # hidden state of the decoder with that of the final hidden states of the encoder.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()

        # embedding dim groups
        self._num_dim_groups = num_dim_groups
        self.dim_names = ["oEffect", "oReact", "oWant", "xAttr", "xEffect",
                          "xIntent", "xNeed", "xReact", "xWant"]
        dim_emb_size = num_dim_groups
        # self.dim_embed = nn.Linear(num_dim_groups,dim_emb_size,bias=False)
        self.dim_embed = lambda x: x

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the

        self.state_decoder = StateDecoderEarlyFusion(
            "decoder", self, num_classes,
            target_embedding_dim, dim_emb_size,
            self._decoder_output_dim
        )
        
        # state_names = [
        #     self.state_name
        # ]

        # self._states: Dict[str, Event2Event.StateDecoder] = {}
        # for name in state_names:
        #     self._states[name] = self.StateDecoder(
        #             name,
        #             self,
        #             num_classes,
        #             target_embedding_dim+num_dim_groups,
        #             self._decoder_output_dim
        #     )

    def _assign_dim_losses(self,dim_array,loss_array):
        dims = dim_array.argmax(dim=1)
        dim_names = self.dim_names
        loss_assign = [dim_names[i] for i in dims]
        loss_dict = {n: 0 for n in dim_names}
        count_dict = {n: 0 for n in dim_names}
        for n,l in zip(loss_assign,loss_array):
            loss_dict[n] += l
            count_dict[n] += 1
        loss_dict = {n: v/count_dict[n] if count_dict[n] else 0 for n,v in loss_dict.items()}
        return loss_dict
        
    def _assign_dim_beams(self,dim_array,all_top_k_predictions):
        # all_top_k_predictions is (bs x beam x seq_len-1)

        dims = dim_array.argmax(dim=1)
        dim_names = self.dim_names
        loss_assign = [dim_names[i] for i in dims]
        loss_dict = {}# {n: [] for n in dim_names}

        for n,l in zip(loss_assign,all_top_k_predictions):
            loss_dict[n] = loss_dict.get(n,[])
            loss_dict[n].append(l)

        loss_dict = {n: torch.stack(v) for n,v in loss_dict.items()}
        assert sum(map(lambda x: x.shape[0], loss_dict.values()))
        return loss_dict
                           
    @overrides
    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                **target_tokens) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the target sequences.

        Parameters
        ----------
        source : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens :
           Dictionary from name to output of ``Textfield.as_array()`` applied on target
           ``TextField``. We assume that the target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        # TODO(brendanr): Revisit dropout.
        embedded_input = self._embedding_dropout(self._source_embedder(source))
        batch_size, _, _ = embedded_input.size()
        source_mask = get_text_field_mask(source)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        output_dict = {}
        
        # Perform greedy search so we can get the loss.
        if target_tokens:
            # if target_tokens.keys() != self._states.keys():
            #     target_only = target_tokens.keys() - self._states.keys()
            #     states_only = self._states.keys() - target_tokens.keys()
            #     raise Exception("Mismatch between target_tokens and self._states. Keys in " +
            #             "targets only: {} Keys in states only: {}".format(target_only, states_only))
            total_loss = 0
            loss_count = 0
            
            targets = target_tokens["target_seq"]#["tokens"]
            dim_arr = target_tokens["dim"]
            emb_dim_arr = self.dim_embed(dim_arr)

            target_sequence_length = targets["tokens"].size()[1]
            if target_sequence_length != 0:
                loss = self.greedy_search(
                    final_encoder_output,
                    targets,
                    self.state_decoder._embedder,
                    self.state_decoder._decoder_cell,
                    self.state_decoder._output_projection_layer,
                    early_fusion=emb_dim_arr,
                    batch_average_loss=False
                )
                loss_dict = self._assign_dim_losses(target_tokens["dim"],loss)
                
                loss = loss.mean()
                total_loss += loss.mean()
                loss_count = loss_count + 1
                output_dict["loss_sum"] = loss.mean()
                output_dict.update({
                    n+"_loss":v for n,v in loss_dict.items()
                })
            
            # Average loss for interpretability.
            if loss_count == 0:
                output_dict["loss"] = 1.0
            else:
                output_dict["loss"] = total_loss / loss_count

        # Perform beam search to obtain the predictions.
        if not self.training:
            # this is true during dev and during generation
            # TODO(Maarten): for generation, target_tokens is {}
            # need to loop over self.dim_names and manually create
            # 9 dim_arr to get all possible dim generations
            if not target_tokens:
                # during generation
                batch_size = self._num_dim_groups
                source_t = source["tokens"].expand(batch_size,source["tokens"].size(1))
                final_encoder_output = final_encoder_output.expand(
                    batch_size,final_encoder_output.size(1))
                source_mask = source_mask.expand(
                    batch_size,source_mask.size(1))
                # manually decoder the sequence 9 times
                dim_arr = torch.eye(self._num_dim_groups)
                emb_dim_arr = self.dim_embed(dim_arr)
                
            (all_top_k_predictions, log_probabilities) = self.beam_search(
                final_encoder_output,
                10,
                self._get_num_decoding_steps(target_tokens.get("target_seq")),
                batch_size,
                source_mask,
                self.state_decoder._embedder,
                self.state_decoder._decoder_cell,
                self.state_decoder._output_projection_layer,
                early_fusion=emb_dim_arr
            )
            #assign beams & probs to the right dimensions
            dim2preds = self._assign_dim_beams(dim_arr, all_top_k_predictions)
            dim2probs = self._assign_dim_beams(dim_arr, log_probabilities)
            
            if target_tokens:
                for dim, top_k_preds in dim2preds.items():
                    self._update_recall(top_k_preds,
                                        target_tokens["target_seq"],
                                        self.state_decoder._recalls[dim])
                
                # ip_embed();exit()
            for dim, top_k_preds in dim2preds.items():
                output_dict["{}_top_k_predictions".format(dim)] = top_k_preds
                output_dict["{}_top_k_log_probabilities".format(dim)] = dim2probs[dim]

        return output_dict
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        # Recall@10 needs beam search which doesn't happen during training.
        if not self.training:
            # ip_embed();exit()
            for dim in self.dim_names:
                all_metrics[dim+"_rec"] = self.state_decoder._recalls[dim].get_metric(reset=reset)
            # all_metrics["target_seq"] = self.state_decoder._recall.get_metric(reset=reset)
        return all_metrics

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds fields for the tokens to the ``output_dict``.
        """
        for name in self.dim_names:
            top_k_predicted_indices = output_dict["{}_top_k_predictions".format(name)][0]
            output_dict["{}_top_k_predicted_tokens".format(name)] = [self.decode_all(top_k_predicted_indices)]
            
        return output_dict


@Model.register("event2event_splitencodedstate")
class Event2Event_SplitEncodedState(Event2Event):
    """
    This ``Event2EventSharedRep`` class is a :class:`Model` which takes an event
    sequence, encodes it, and then uses the encoded representation to decode
    several mental state sequences.

    See: https://www.semanticscholar.org/paper/Event2Mind/b89f8a9b2192a8f2018eead6b135ed30a1f2144d

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : int, required
        Length of decoded sequences
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 decoder_hidden_size: int = None,
                 target_fields = None,
                 target_fields_categorization = None,
                 metrics = None) -> None:
        super(Event2Event, self).__init__(vocab)
        # TODO(brendanr): Hack the embeddings here like initWEmb in modeling/utils/preprocess.py?
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._embedding_dropout = nn.Dropout(0.2)
        self._target_fields = target_fields

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder.
        self._encoder_output_dim = self._encoder.get_output_dim()
        self._decoder_output_dim = decoder_hidden_size or self._encoder_output_dim
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()

        self._states: Dict[str, Event2Event.StateDecoder] = {}
        self._states_assignments = target_fields_categorization.as_dict(quiet=True)

        # {
        #     "xNeed": "pre",
        #     "xIntent": "pre",
        #     "xAttr": "pre",
        #     'oEffect': "post",
        #     'oReact': "post",
        #     'oWant': "post",
        #     'xEffect': "post",
        #     'xReact': "post",
        #     'xWant': "post"
        # }
        count = Counter(self._states_assignments.values())
        
        self._dim_splits = OrderedDict()
        offset = 0
        for i, (cat, c) in enumerate(count.items()):
            dim_size = int(self._encoder_output_dim * c / len(self._states_assignments))
            self._dim_splits[cat] = [offset, dim_size+offset]
            offset += dim_size
        
        for field in self._target_fields:
            b,e = self._dim_splits[self._states_assignments[field]]
            init_dim = e-b
            self._states[field] = StateDecoderLinear(
                name=field, event2event=self, num_classes=num_classes,
                input_dim=target_embedding_dim, init_dim=init_dim,
                output_dim=self._decoder_output_dim,
                metrics=metrics, slices=(b,e)
            )

