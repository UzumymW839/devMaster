import unittest
import torch

import losses
from unittests.utils_for_testing import all_subclasses


class TestLosses(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        base_class = losses.BaseLoss
        self.classes_to_test = all_subclasses(base_class)

        self.expected_shape = (4, 1, 64000)


    def test_perfect_prediction(self):
        random_predicted = torch.randn(size=self.expected_shape)
        random_target = random_predicted.clone()

        for loss_class in self.classes_to_test:
            loss = loss_class()
            out_value = loss(predicted=random_predicted, target=random_target)

            self.assertAlmostEqual(out_value, loss_class._perfect_score, places=2,
                                msg=f'Loss of class {loss_class} should have an output of {loss_class._perfect_score} for perfect input, returned {out_value} instead!')


    def test_degraded_score(self):
        random_predicted = torch.randn(size=self.expected_shape)
        random_target = random_predicted.clone()
        random_degraded = random_target.clone() + torch.randn_like(random_target)

        for loss_class in self.classes_to_test:
            loss = loss_class()
            out_perfect = loss(predicted=random_predicted, target=random_target)
            out_degraded = loss(predicted=random_degraded, target=random_target)

            self.assertLess(out_perfect, out_degraded) # larger loss when degraded than perfect





if __name__ == '__main__':
    unittest.main()
