from overrides import overrides

from allennlp.training.metrics.metric import Metric


@Metric.register("average")
class Average(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value += list(self.unwrap_to_tensors(value))[0]
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = float(self._total_value) / float(self._count) if self._count > 0 else 0.
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0

@Metric.register("batched_average")
class BatchedAverage(Average):
    def __call__(self, val_sum, count):
        self._total_value += list(self.unwrap_to_tensors(val_sum))[0]
        self._count += list(self.unwrap_to_tensors(count))[0]

