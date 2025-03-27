import unittest
import torch

import metrics
from unittests.utils_for_testing import all_subclasses


class TestMetrics(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        base_class = metrics.BaseMetric
        self.classes_to_test = all_subclasses(base_class)

        self.expected_shape = (1, 64000)
        self.test_device = torch.device('cpu')



    def test_perfect_prediction(self):
        random_predicted = torch.randn(size=self.expected_shape)
        random_target = random_predicted.clone()

        for metric_class in self.classes_to_test:
            met = metric_class(device=self.test_device)
            out_value = met(predicted=random_predicted, target=random_target)

            if metric_class == metrics.SISDRMetric:
                self.assertGreater(out_value, metric_class._perfect_score)
            else:
                self.assertAlmostEqual(out_value, metric_class._perfect_score, places=2,
                                    msg=f'metric of class {metric_class} should have an output of {metric_class._perfect_score} for perfect input, returned {out_value} instead!')


    def test_degraded_score(self):
        random_predicted = torch.randn(size=self.expected_shape)
        random_target = random_predicted.clone()
        random_degraded = random_target.clone() + torch.randn_like(random_target)

        for metric_class in self.classes_to_test:
            met = metric_class(device=self.test_device)
            out_perfect = met(predicted=random_predicted, target=random_target)
            out_degraded = met(predicted=random_degraded, target=random_target)

            if metric_class._improvement_higher_scores:
                self.assertGreater(out_perfect, out_degraded)
            else:
                self.assertLess(out_perfect, out_degraded)





if __name__ == '__main__':
    unittest.main()
