# pylint: disable=no-self-use,invalid-name,protected-access
import torch
<<<<<<< HEAD
import pytest
import numpy

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
=======
import numpy

from allennlp.common.testing import AllenNlpTestCase
>>>>>>> 51d20c54cdbb147f63285b4e20a3ab5b6ba5fa73
from allennlp.training.metrics import SequenceAccuracy

class SequenceAccuracyTest(AllenNlpTestCase):
    def test_sequence_accuracy(self):
        accuracy = SequenceAccuracy()
<<<<<<< HEAD
        gold = torch.tensor([
            [1, 2, 3],
            [2, 4, 8],
            [0, 1, 1]
        ])
        predictions = torch.tensor([
            [[1, 2, 3], [1, 2, -1]],
            [[2, 4, 8], [2, 5, 9]],
            [[-1, -1, -1], [0, 1, -1]]
=======
        gold = torch.Tensor([
                [1, 2, 3],
                [2, 4, 8],
                [0, 1, 1]
        ])
        predictions = torch.Tensor([
                [[1, 2, 3], [1, 2, -1]],
                [[2, 4, 8], [2, 5, 9]],
                [[-1, -1, -1], [0, 1, -1]]
>>>>>>> 51d20c54cdbb147f63285b4e20a3ab5b6ba5fa73
        ])

        accuracy(predictions, gold)
        actual_accuracy = accuracy.get_metric()
        numpy.testing.assert_almost_equal(actual_accuracy, 2/3)

    def test_sequence_accuracy_respects_mask(self):
        accuracy = SequenceAccuracy()
<<<<<<< HEAD
        gold = torch.tensor([
            [1, 2, 3],
            [2, 4, 8],
            [0, 1, 1],
            [11, 13, 17],
        ])
        predictions = torch.tensor([
            [[1, 2, 3], [1, 2, -1]],
            [[2, 4, 8], [2, 5, 9]],
            [[-1, -1, -1], [0, 1, -1]],
            [[12, 13, 17], [11, 13, 18]]
        ])
        mask = torch.tensor([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1]
=======
        gold = torch.Tensor([
                [1, 2, 3],
                [2, 4, 8],
                [0, 1, 1],
                [11, 13, 17],
        ])
        predictions = torch.Tensor([
                [[1, 2, 3], [1, 2, -1]],
                [[2, 4, 8], [2, 5, 9]],
                [[-1, -1, -1], [0, 1, -1]],
                [[12, 13, 17], [11, 13, 18]]
        ])
        mask = torch.Tensor([
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 0, 1]
>>>>>>> 51d20c54cdbb147f63285b4e20a3ab5b6ba5fa73
        ])

        accuracy(predictions, gold, mask)
        actual_accuracy = accuracy.get_metric()
        numpy.testing.assert_almost_equal(actual_accuracy, 3/4)

    def test_sequence_accuracy_accumulates_and_resets_correctly(self):
        accuracy = SequenceAccuracy()
<<<<<<< HEAD
        gold = torch.tensor([
            [1, 2, 3],
        ])
        accuracy(torch.tensor([[[1, 2, 3]]]), gold)
        accuracy(torch.tensor([[[1, 2, 4]]]), gold)
=======
        gold = torch.Tensor([[1, 2, 3]])
        accuracy(torch.Tensor([[[1, 2, 3]]]), gold)
        accuracy(torch.Tensor([[[1, 2, 4]]]), gold)
>>>>>>> 51d20c54cdbb147f63285b4e20a3ab5b6ba5fa73

        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 1/2)
        assert accuracy.correct_count == 0
        assert accuracy.total_count == 0
<<<<<<< HEAD
=======

    def test_get_metric_on_new_object_works(self):
        accuracy = SequenceAccuracy()

        actual_accuracy = accuracy.get_metric(reset=True)
        numpy.testing.assert_almost_equal(actual_accuracy, 0)
>>>>>>> 51d20c54cdbb147f63285b4e20a3ab5b6ba5fa73
