import sys
import torch
from typing import Optional
from overrides import overrides
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

from nltk import bleu

from IPython import embed as ip_embed

@Metric.register("bleu_n")
class BleuN(Metric):
    """
    BLEU metric
    """
    def __init__(self, n: int = 4) -> None:
        self.n = n
        self.weights = [1/n] * n
        self.bl_total = 0.
        self.count = 0

    def _scoring_f(self, hyp, refs):
        return bleu(refs,hyp,weights=self.weights)
    
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 end_index: int = sys.maxsize):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, k, sequence_length).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.dim() - 1 but "
                                     "found tensor of shape: {}".format(gold_labels.size()))
        if mask is not None and mask.size() != gold_labels.size():
            raise ConfigurationError("mask must have the same size as predictions but "
                                     "found tensor of shape: {}".format(mask.size()))

        k = predictions.size()[1]
        batch_size = predictions.size()[0]
        
        bl_scores = []
        
        for i, (beams, cur_gold) in enumerate(zip(predictions,gold_labels)):
            if mask is not None:
                masked_gold = cur_gold * mask[i]
                masked_beams = [b*mask[i] for b in beams]
            else:
                masked_gold = cur_gold
                masked_beams = beams

            # HACK: turn tensors into strings cause nltk.bleu() doesn't work with tensors (only str or int)
            cleaned_gold = [str(x) for x in masked_gold if x != 0 and x != end_index]
            cleaned_beams = [[str(x) for x in b if x != 0 and x != end_index]
                             for b in masked_beams]
            bl = self._scoring_f(cleaned_gold,cleaned_beams)
            bl_scores.append(bl)
            
        self.bl_total += sum(bl_scores)
        self.count += len(bl_scores)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated rouge score.
        """
        bleu_score = float(self.bl_total) / float(self.count)
        if reset:
            self.reset()
        return bleu_score

    @overrides
    def reset(self):
        self.bl_total = 0.
        self.count = 0

