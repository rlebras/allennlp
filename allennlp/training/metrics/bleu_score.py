import sys
import torch
from typing import Optional
from overrides import overrides
from allennlp.common.checks import ConfigurationError

from allennlp.data.tokenizers.token import Token
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics.metric import Metric

from nltk.translate.bleu_score import SmoothingFunction
from nltk import bleu

from IPython import embed as ip_embed

@Metric.register("bleu_n")
class BleuN(Metric):
    """
    BLEU metric
    """
    def __init__(self, n: int = 2) -> None:
        self.n = n
        self.weights = [1/n] * n
        self.bl_total = 0.
        self.count = 0

    def _scoring_f(self, hyp, refs):
        return bleu(refs,hyp,weights=self.weights,smoothing_function=SmoothingFunction().method1)
    
    def __call__(self,
                 references: torch.Tensor,
                 hypotheses: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 end_index: int = sys.maxsize,
                 none_index: int = None,
                 dont_count_empty_predictions = False):
        """
        Parameters
        ----------
        references : ``torch.Tensor``, required.
            A tensor of integer class labels of shape (N, sequence_length) where N is the number of references.
        hypotheses : ``torch.Tensor``, required.
            A tensor of integer class label of shape (K, sequence_length) where K is the number of hypotheses. 
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        
        hypotheses, references, mask = self.unwrap_to_tensors(hypotheses, references, mask)

        # Some sanity checks.
        if references.dim() != hypotheses.dim():
            raise ConfigurationError("references must have dimension == predictions.dim() but "
                                     "found tensor of shape: {}".format(references.size()))
        if mask is not None and mask.size() != references.size():
            raise ConfigurationError("mask must have the same size as predictions but "
                                     "found tensor of shape: {}".format(mask.size()))

        k = hypotheses.size()[0]
        batch_size = 1 # .size()[0]
        
        bl_scores = []

        # if mask, gold needs to do something?
        
        # truncating references to the end_token
        clean_refs = [[str(x.item()) for x in ref if x.item() != 0 and x.item() != end_index]
                      for ref in references]

        for i, beam in enumerate(hypotheses):
            if mask is not None:
                masked_beams = [b*mask[i] for b in beams]
            else:
                masked_beam = beam

            # HACK: turn tensors into strings cause nltk.bleu() doesn't work with tensors (only str or int)
            # cleaned_gold = [str(x.item()) for x in masked_gold if x.item() != 0 and x.item() != end_index]
            # cleaned_beams = [[str(x.item()) for x in b if x.item() != 0 and x.item() != end_index]
            #                  for b in masked_beams]
            clean_beam = [str(x.item()) for x in masked_beam if x.item() != 0 and x.item() != end_index]
            
            if none_index is not None and sum([x == [str(none_index)] for x in clean_refs])/len(clean_refs) > 1/3:
                # Oracle about Nones
                
                # bl = 1
                # bl_scores.append(bl)
                
                # if sum([x == [str(none_index)] for x in clean_refs])/len(clean_refs) > 1/3:
                #     ip_embed();exit()
                continue

            #print("cleaned_gold: ", cleaned_gold)
            #print("cleaned_beams: ", cleaned_beams)
            if not dont_count_empty_predictions or not clean_beam == []:
                bl = self._scoring_f(clean_beam,clean_refs)
                bl_scores.append(bl)
                
            # if dont_count_empty_predictions and not all(x == [] for x in cleaned_beams):
            #     bl = self._scoring_f(cleaned_gold, cleaned_beams)
            #     bl_scores.append(bl)
            # else:
            #     bl = self._scoring_f(cleaned_gold,cleaned_beams)
            #     bl_scores.append(bl)

        #raise Exception('STOP!')
        self.bl_total += sum(bl_scores)
        self.count += len(bl_scores)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated bleu score.
        """
        if self.count != 0:
            bleu_score = float(self.bl_total) / float(self.count)
        else:
            bleu_score = 0.
        if reset:
            self.reset()
        return bleu_score

    @overrides
    def reset(self):
        self.bl_total = 0.
        self.count = 0

