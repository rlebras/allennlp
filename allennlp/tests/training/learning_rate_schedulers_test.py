# pylint: disable=no-self-use,invalid-name,protected-access

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler, CosineWithRestarts
from allennlp.common.params import Params


class LearningRateSchedulersTest(AllenNlpTestCase):

    def test_reduce_on_plateau_error_throw_when_no_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        with self.assertRaises(ConfigurationError) as context:
            LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                    Params({"type": "adam"})),
                                              Params({"type": "reduce_on_plateau"})).step(None, None)

        self.assertTrue(
                'The reduce_on_plateau learning rate scheduler requires a validation metric'
                in str(context.exception))

    def test_reduce_on_plateau_works_when_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                Params({"type": "adam"})),
                                          Params({"type": "reduce_on_plateau"})).step(10, None)

    def test_no_metric_wrapper_can_support_none_for_metrics(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        lrs = LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                      Params({"type": "adam"})),
                                                Params({"type": "step", "step_size": 1}))
        lrs.step(None, None)

    def test_noam_learning_rate_schedule_does_not_crash(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        lrs = LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                      Params({"type": "adam"})),
                                                Params({"type": "noam", "model_size": 10, "warmup_steps": 2000}))
        lrs.step(None)
        lrs.step_batch(None)


class CosineWithRestartsTest(AllenNlpTestCase):

    def setUp(self):
        super(CosineWithRestartsTest, self).setUp()
        self.model = torch.nn.Sequential(torch.nn.Linear(10, 10))

    def _get_optimizer(self, lr: float = 1.0):
        return Optimizer.from_params(self.model.named_parameters(), Params({"type": "sgd", "lr": lr}))

    def test_from_params(self):
        optim = self._get_optimizer()
        sched = LearningRateScheduler.from_params(optim, Params({"type": "cosine", "t_max": 5})).lr_scheduler

        assert sched.t_max == 5
        assert sched._initialized is True

        # Learning should be unchanged after initializing scheduler.
        assert optim.param_groups[0]["lr"] == 1.0

        with self.assertRaises(TypeError):
            # t_max is required.
            LearningRateScheduler.from_params(optim, Params({"type": "cosine"}))

    def test_schedules(self):
        cosine_schedule_cases = [
                (30, 30, 1.0, [(0, 1.0),
                               (15, 0.5000000000000001),
                               (29, 0.0027390523158632996)]),
                (10, 1, 2.0, [(0, 1.0),
                              (1, 1.0),
                              (2, 0.5),
                              (3, 1.0)]),
                (30, 1, 1.0, [(0, 1.0),
                              (15, 1.0),
                              (29, 1.0)]),
                (60, 30, 1.0, [(0, 1.0),
                               (15, 0.5000000000000001),
                               (29, 0.0027390523158632996),
                               (30, 1.0),
                               (45, 0.5000000000000001),
                               (59, 0.0027390523158632996)]),
                (100, 30, 1.5, [(0, 1.0),
                                (29, 0.0027390523158632996),
                                (30, 1.0),
                                (74, 0.0012179748700879012)]),
                (210, 30, 2, [(0, 1.0),
                              (29, 0.0027390523158632996),
                              (30, 1.0),
                              (89, 0.0006852326227130834),
                              (90, 1.0),
                              (209, 0.00017133751222137006)]),
                (150, 30, 1, [(0, 1.0),
                              (29, 0.0027390523158632996),
                              (30, 1.0),
                              (59, 0.0027390523158632996),
                              (60, 1.0),
                              (89, 0.0027390523158632996),
                              (90, 1.0)]),
        ]

        for epochs, t_max, factor, lr_checks in cosine_schedule_cases:
            optimizer = self._get_optimizer()
            scheduler = CosineWithRestarts(optimizer, t_max, factor=factor)
            lrs = [optimizer.param_groups[0]["lr"]]
            for epoch in range(epochs):
                scheduler.step(epoch)
                lrs.append(optimizer.param_groups[0]["lr"])

            for it, lr in lr_checks:
                assert lrs[it] == lr
