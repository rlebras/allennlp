from typing import Optional
from overrides import overrides
import sys
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
import collections

import six

from IPython import embed as ip_embed
      
@Metric.register("rouge_l")
class RougeL(Metric):
    """
    ROUGE-L metric. (based on longest common subsequences)
    """
    def __init__(self, alpha: float = .5) -> None:
        self.alpha = alpha
        self.rg_total = 0.
        self.count = 0

    def _scoring_f(self, hyp, refs):
        return rouge_l(hyp,refs,self.alpha)
    
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 end_index: int = sys.maxsize,
                 dont_count_empty_predictions = False):
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
        
        rg_scores = []
        
        for i, (beams, cur_gold) in enumerate(zip(predictions,gold_labels)):
            if mask is not None:
                masked_gold = cur_gold * mask[i]
                masked_beams = [b*mask[i] for b in beams]
            else:
                masked_gold = cur_gold
                masked_beams = beams
                
            cleaned_gold = [x.item() for x in masked_gold if x.item() != 0 and x.item() != end_index]
            cleaned_beams = [[x.item() for x in b if x.item() != 0 and x.item() != end_index]
                             for b in masked_beams]
            if dont_count_empty_predictions:
                all_empty = all(x == [] for x in cleaned_beams)
                if not all_empty:
                    rg = self._scoring_f(cleaned_gold, cleaned_beams)
                    rg_scores.append(rg)
            else:
                rg = self._scoring_f(cleaned_gold,cleaned_beams)
                rg_scores.append(rg)
        self.rg_total += sum(rg_scores)
        self.count += len(rg_scores)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated rouge score.
        """
        if self.count != 0:
            rouge = float(self.rg_total) / float(self.count)
        else:
            rouge = 0.
        if reset:
            self.reset()
        return rouge

    @overrides
    def reset(self):
        self.rg_total = 0.
        self.count = 0
            
@Metric.register("rouge_n")
class RougeN(RougeL):
    """
    ROUGE-N metric.
    """
    def __init__(self, alpha: float = .5, n: int = 2) -> None:
        self.n = n
        self.alpha = alpha
        self.rg_total = 0.
        self.count = 0
        
    def _scoring_f(self, hyp, refs):
        return rouge_n(hyp,refs,self.n,self.alpha)

def _ngrams(words, n):
    queue = collections.deque(maxlen=n)
    for w in words:
        queue.append(w)
        if len(queue) == n:
            yield tuple(queue)

def _ngram_counts(words, n):
    return collections.Counter(_ngrams(words, n))

def _ngram_count(words, n):
    return max(len(words) - n + 1, 0)

def _counter_overlap(counter1, counter2):
    result = 0
    for k, v in six.iteritems(counter1):
        result += min(v, counter2[k])
    return result

def _safe_divide(numerator, denominator):
    if denominator > 0:
        return numerator / denominator
    else:
        return 0

def _safe_f1(matches, recall_total, precision_total, alpha):
    recall_score = _safe_divide(matches, recall_total)
    precision_score = _safe_divide(matches, precision_total)
    denom = (1.0 - alpha) * precision_score + alpha * recall_score
    if denom > 0.0:
        return (precision_score * recall_score) / denom
    else:
        return 0.0

def rouge_n(peer, models, n, alpha):
    """
    Compute the ROUGE-N score of a peer with respect to one or more models, for
    a given value of `n`.
    """
    matches = 0
    recall_total = 0
    peer_counter = _ngram_counts(peer, n)
    for model in models:
        model_counter = _ngram_counts(model, n)
        matches += _counter_overlap(peer_counter, model_counter)
        recall_total += _ngram_count(model, n)
    precision_total = len(models) * _ngram_count(peer, n)
    return _safe_f1(matches, recall_total, precision_total, alpha)

def rouge_1(peer, models, alpha):
    """
    Compute the ROUGE-1 (unigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 1, alpha)

def rouge_2(peer, models, alpha):
    """
    Compute the ROUGE-2 (bigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 2, alpha)

def rouge_3(peer, models, alpha):
    """
    Compute the ROUGE-3 (trigram) score of a peer with respect to one or more
    models.
    """
    return rouge_n(peer, models, 3, alpha)

def lcs(a, b):
    """
    Compute the length of the longest common subsequence between two sequences.

    Time complexity: O(len(a) * len(b))
    Space complexity: O(min(len(a), len(b)))
    """
    # This is an adaptation of the standard LCS dynamic programming algorithm
    # tweaked for lower memory consumption.
    # Sequence a is laid out along the rows, b along the columns.
    # Minimize number of columns to minimize required memory
    if len(a) < len(b):
        a, b = b, a
    # Sequence b now has the minimum length
    # Quit early if one sequence is empty
    if len(b) == 0:
        return 0
    # Use a single buffer to store the counts for the current row, and
    # overwrite it on each pass
    row = [0] * len(b)
    for ai in a:
        left = 0
        diag = 0
        for j, bj in enumerate(b):
            up = row[j]
            if ai == bj:
                value = diag + 1
            else:
                value = max(left, up)
            row[j] = value
            left = value
            diag = up
    # Return the last cell of the last row
    return left

def rouge_l(peer, models, alpha):
    """
    Compute the ROUGE-L score of a peer with respect to one or more models.
    """
    matches = 0
    recall_total = 0
    for model in models:
        matches += lcs(model, peer)
        recall_total += len(model)
    precision_total = len(models) * len(peer)
    return _safe_f1(matches, recall_total, precision_total, alpha)
