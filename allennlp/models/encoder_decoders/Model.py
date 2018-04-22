from typing import Dict

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from overrides import overrides
from torch.autograd import Variable
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common import Params
from allennlp.data.dataset_readers.seq2seq import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from allennlp.type_checking import valid_next_characters, update_state
import random

"""
And (bool, ...) : bool
Equals (num, num) : bool
Join   (set, set) : set
Value  (set, type) : num
Rate   (set, type, type) : num
Minus  (num, num) : num
Plus   (num, num) : num


"""


def batched_index_select(input_tensor, index_tensor):
    dummy = index_tensor.unsqueeze(1).unsqueeze(2).expand(index_tensor.size(0), 1,
                                                          input_tensor.size(2))
    out = input_tensor.gather(1, dummy)  # b x e x f
    return out.squeeze(dim=1)


@Model.register("simple_copy")
class SimpleCopy(Model):
    """
    This ``SimpleSeq2Seq`` class is a :class:`Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    This ``SimpleSeq2Seq`` model takes an encoder (:class:`Seq2SeqEncoder`) as an input, and
    implements the functionality of the decoder.  In this implementation, the decoder uses the
    encoder's outputs in two ways. The hidden state of the decoder is initialized with the output
    from the final time-step of the encoder, and when using attention, a weighted average of the
    outputs from the encoder is concatenated to the inputs of the decoder at every timestep.

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
    attention_function: ``SimilarityFunction``, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    scheduled_sampling_ratio: float, optional (default = 0.0)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 max_decoding_steps: int,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None,
                 attention_function: SimilarityFunction = None,
                 scheduled_sampling_ratio: float = 0.0) -> None:
        super(SimpleCopy, self).__init__(vocab)
        self._source_embedder = source_embedder
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace
        self._attention_function = attention_function
        self._scheduled_sampling_ratio = scheduled_sampling_ratio
        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        self.num_classes = num_classes
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = self._encoder.get_output_dim()
        if self._attention_function:
            self._decoder_attention = Attention(self._attention_function)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time step.
            self._decoder_input_dim = self._encoder.get_output_dim() + target_embedding_dim
        else:
            self._decoder_input_dim = target_embedding_dim
        # TODO (pradeep): Do not hardcode decoder cell type.
        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_output_dim)
        self._output_embeddings = torch.nn.Parameter(
            torch.randn(num_classes, target_embedding_dim) / 10)
        self._random_embedding_size = 50
        self._special_tokens = torch.nn.Parameter(torch.randn(3, self._random_embedding_size) / 10)
        self._stem_scale = torch.nn.Parameter(torch.randn(1, 1) / 10)
        # self._stem_embedding = torch.nn.Parameter(torch.randn(251, self._random_embedding_size))
        # self._permutable_indices = list(range(3, 251))


    def beam_search(self,  # type: ignore
                    source_tokens: Dict[str, torch.LongTensor],
                    bestk: int) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        source_indices = source_tokens['tokens'].data.cpu().numpy()
        embedded_input = self._source_embedder(source_tokens)
        batch_size, _, _ = embedded_input.size()
        assert batch_size == 1, batch_size
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        decoder_hidden = final_encoder_output
        decoder_context = Variable(encoder_outputs.data.new()
                                   .resize_(batch_size, self._decoder_output_dim).fill_(0))
        # For each action I should keep around a score, a context and a hidden state, and input choices
        state = source_mask.data.new().resize_(batch_size)
        model = {
            'last_prediction': self._start_index,
            'decoder_hidden': decoder_hidden,
            'decoder_context': decoder_context,
            'cur_log_probability': 0,
            'action_list': [START_SYMBOL],
            'arg_numbers': [0],
            'function_calls': []
        }
        valid_variables = {'var' + str(i) for i in range(20)}
        valid_variables.add('(')
        valid_variables.add('?')
        valid_units = {'unit' + str(i) for i in range(20)}
        valid_numbers = {self.vocab.get_token_from_index(index, 'source_tokens') for index in
                         source_indices[0]}
        valid_numbers = {x for x in valid_numbers if x.startswith('num')}
        # print(valid_numbers)
        valid_numbers.add('(')
        valid_numbers.add('?')
        models = [model]
        complete_models = []
        for cur_length in range(self._max_decoding_steps + 2):
            new_models = []
            for model in models:
                last_prediction = model['last_prediction']
                action_list = model['action_list']
                if action_list[-1] == END_SYMBOL:
                    new_models.append(model)
                    continue
                assert len(action_list) == cur_length + 1, (len(action_list), cur_length + 1)
                decoder_hidden = model['decoder_hidden']
                decoder_context = model['decoder_context']
                cur_log_probability = model['cur_log_probability']
                decoder_input = self._prepare_decode_step_input(
                    Variable(state.fill_(last_prediction)),
                    decoder_hidden,
                    encoder_outputs,
                    source_mask
                )
                print(decoder_input.size())
                decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                     (decoder_hidden,
                                                                      decoder_context))
                output_projections = self._output_projection_layer(decoder_hidden)
                class_log_probabilities = \
                    F.log_softmax(output_projections, dim=-1).data.cpu().numpy()[0]
                assert self.vocab.get_vocab_size(self._target_namespace) == len(
                    class_log_probabilities), (self.vocab.get_vocab_size(self._target_namespace),
                                               class_log_probabilities.shape[0])
                decoded_new_variable = False
                decoded_new_unit = False
                valid_actions = valid_next_characters(model['function_calls'],
                                                      model['arg_numbers'],
                                                      action_list[-1],
                                                      valid_numbers,
                                                      valid_variables,
                                                      valid_units)
                for action_index, action_log_probability in enumerate(class_log_probabilities):
                    action = self.vocab.get_token_from_index(action_index, self._target_namespace)
                    if action not in valid_actions:
                        continue
                    # if action.startswith('var'):
                    #     seen = action in action_list
                    #     # If a variable is new, we do not distinguish between which variable it is
                    #     if not seen and decoded_new_variable:
                    #         continue
                    #     elif not seen:
                    #         decoded_new_variable = True

                    function_calls, arg_numbers = update_state(model['function_calls'],
                                                               model['arg_numbers'], action)
                    new_model = {
                        'action_list': action_list + [action],
                        'last_prediction': action_index,
                        'decoder_hidden': decoder_hidden,
                        'decoder_context': decoder_context,
                        'cur_log_probability': action_log_probability + cur_log_probability,
                        'function_calls': function_calls,
                        'arg_numbers': arg_numbers
                    }
                    new_models.append(new_model)
            new_models.sort(key=lambda x: - x['cur_log_probability'])
            models = new_models[:bestk]

        complete_models = [model for model in models if model['action_list'][-1] == END_SYMBOL]
        complete_models.sort(key=lambda x: - x['cur_log_probability'])
        # print('total models', len(models), 'len complete models', len(complete_models))
        output = '\n'.join([' '.join(model['action_list'][1:-1]) for model in complete_models]) + \
                 '\n***\n'
        # print(' '.join(complete_models[0]['action_list'][1:-1]))
        return output

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                stem_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : Dict[str, torch.LongTensor]
           The output of ``TextField.as_array()`` applied on the source ``TextField``. This will be
           passed through a ``TextFieldEmbedder`` and then through an encoder.
        target_tokens : Dict[str, torch.LongTensor], optional (default = None)
           Output of ``Textfield.as_array()`` applied on target ``TextField``. We assume that the
           target tokens are also represented as a ``TextField``.
           stem_tokens:
        """
        # (batch_size, input_sequence_length, encoder_output_dim)
        embedded_input = self._source_embedder(source_tokens)

        stem_tokens = stem_tokens['tokens']
        batch_size, num_timesteps, original_embedding_dim = embedded_input.size()
        # random.shuffle(self._permutable_indices)
        random_vocab = torch.randn(num_timesteps, self._random_embedding_size)
        random_vocab = torch.autograd.Variable(random_vocab, requires_grad=False)
        random_vocab = random_vocab.cuda() * self._stem_scale.cuda()
        random_vocab = torch.cat([self._special_tokens, random_vocab], dim=0)
        flattened_indices = stem_tokens.view(stem_tokens.numel())
        random_embeddings = torch.index_select(random_vocab, 0, flattened_indices)
        # (batch_size, nm_timesteps, embedding_dim)
        random_embeddings = random_embeddings.view((batch_size,
                                                    num_timesteps,
                                                    self._random_embedding_size))
        embedded_input = torch.cat([embedded_input, random_embeddings], dim=2)
        # assert batch_size == 1
        source_mask = get_text_field_mask(source_tokens)
        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = encoder_outputs[:, -1]  # (batch_size, encoder_output_dim)
        if target_tokens:
            targets = target_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end symbol. Either way, we
            # don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps
        decoder_hidden = final_encoder_output
        decoder_context = Variable(encoder_outputs.data.new()
                                   .resize_(batch_size, self._decoder_output_dim).fill_(0))
        last_predictions = None
        step_logits = []
        step_probabilities = []
        step_predictions = []
        # encoder_outputs has shape (batch size, num time steps, embedding dim)
        # self._output_embeddings needs to be expanded to (batch size, num actions, embedding dim)
        # print(self._output_embeddings.data.cpu().numpy())
        # print(encoder_outputs.data.cpu().numpy())
        basic_actions = self._output_embeddings.unsqueeze(0).expand((batch_size, -1, -1))
        output_embeddings = torch.cat([basic_actions, encoder_outputs], dim=1)
        # output_embeddings should have shape (batch size, num actions + num time steps, embedding dim)
        for timestep in range(num_decoding_steps):
            if self.training:
                input_choices = targets[:, timestep]
            else:
                if timestep == 0:
                    # For the first timestep, when we do not have targets, we input start symbols.
                    # (batch_size,)
                    input_choices = Variable(source_mask.data.new()
                                             .resize_(batch_size).fill_(self._start_index))
                else:
                    input_choices = last_predictions
            decoder_input = self._prepare_decode_step_input(input_choices,
                                                            output_embeddings,
                                                            decoder_hidden,
                                                            encoder_outputs,
                                                            source_mask)
            decoder_hidden, decoder_context = self._decoder_cell(decoder_input,
                                                                 (decoder_hidden, decoder_context))
            output_logits = (output_embeddings * decoder_hidden.unsqueeze(1)).sum(dim=-1)

            # output_logits (batch size, num actions + num tokens)
            class_probabilities = torch.nn.functional.softmax(output_logits, dim=-1)
            step_logits.append(output_logits.unsqueeze(1))
            _, predicted_classes = torch.max(class_probabilities, 1)
            step_probabilities.append(class_probabilities.unsqueeze(1))
            last_predictions = predicted_classes
            step_predictions.append(last_predictions.unsqueeze(1))
        # step_logits is a list containing tensors of shape (batch_size, 1, num_classes)
        # This is (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        class_probabilities = torch.cat(step_probabilities, 1)
        all_predictions = torch.cat(step_predictions, 1)
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "predictions": all_predictions}
        if target_tokens:
            target_mask = get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss
            # TODO: Define metrics
            # if random.random() < 0.01:
            #     print('\naccuracy',
            #           self._get_accuracy(all_predictions.cpu(), targets.cpu(), target_mask.cpu()))
        return output_dict

    def _prepare_decode_step_input(self,
                                   input_indices: torch.LongTensor,
                                   embeddings: torch.LongTensor,
                                   decoder_hidden_state: torch.LongTensor = None,
                                   encoder_outputs: torch.LongTensor = None,
                                   encoder_outputs_mask: torch.LongTensor = None) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep. So, this can be used even if we're doing some kind of scheduled sampling.

        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = batched_index_select(embeddings, input_indices)
        if self._attention_function:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.
            encoder_outputs_mask = encoder_outputs_mask.float()
            # (batch_size, input_sequence_length)
            input_weights = self._decoder_attention(decoder_hidden_state, encoder_outputs,
                                                    encoder_outputs_mask)
            # (batch_size, encoder_output_dim)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            # (batch_size, encoder_output_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.LongTensor:
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
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    @staticmethod
    def _get_accuracy(predictions: torch.LongTensor,
                      targets: torch.LongTensor,
                      target_mask: torch.LongTensor) -> torch.LongTensor:
        targets = targets[:, 1:]
        targets = targets.data.numpy()
        target_mask = target_mask[:, 1:]
        target_mask = target_mask.data.numpy()
        predictions = predictions.data.numpy()
        total = np.sum((targets == predictions) * target_mask)
        return total / np.sum(target_mask)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.data.cpu().numpy()
        all_predicted_tokens = []
        for instance_index, indices in enumerate(predicted_indices):
            indices = list(indices)
            # Collect indices till the first end_symbol

            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            else:
                print(indices, self._end_index)
            predicted_tokens = []
            for index in indices:
                if index < self.vocab.get_vocab_size(self._target_namespace):
                    token = self.vocab.get_token_from_index(index,
                                                            namespace=self._target_namespace)
                else:
                    token = str(index - self.vocab.get_vocab_size(self._target_namespace))
                predicted_tokens.append(token)
            all_predicted_tokens.append(predicted_tokens)
        output_dict["predicted_tokens"] = all_predicted_tokens
        return output_dict

    @classmethod
    def from_params(cls, vocab, params: Params) -> 'SimpleCopy':
        source_embedder_params = params.pop("source_embedder")
        source_embedder = TextFieldEmbedder.from_params(vocab, source_embedder_params)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        max_decoding_steps = params.pop("max_decoding_steps")
        target_namespace = params.pop("target_namespace", "tokens")
        # If no attention function is specified, we should not use attention, not attention with
        # default similarity function.
        attention_function_type = params.pop("attention_function", None)
        target_embedding_dim = params.pop("target_embedding_dim", None)
        if attention_function_type is not None:
            attention_function = SimilarityFunction.from_params(attention_function_type)
        else:
            attention_function = None
        scheduled_sampling_ratio = params.pop_float("scheduled_sampling_ratio", 0.0)
        params.assert_empty(cls.__name__)
        return cls(vocab,
                   source_embedder=source_embedder,
                   encoder=encoder,
                   max_decoding_steps=max_decoding_steps,
                   target_namespace=target_namespace,
                   target_embedding_dim=target_embedding_dim,
                   attention_function=attention_function,
                   scheduled_sampling_ratio=scheduled_sampling_ratio)
