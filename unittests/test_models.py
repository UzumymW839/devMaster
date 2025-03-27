import unittest
import torch

import models
from unittests.utils_for_testing import all_subclasses

class TestModels(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        base_class = models.BaseModel
        self.model_classes_to_test = all_subclasses(base_class)

        self.expected_input_shape = (1, 2, 32000)
        self.expected_output_shape = (1, 1, 32000)
        self.test_device = torch.device('cpu')


    def test_model_construction(self):
        for expected_model_class in self.model_classes_to_test:

            if issubclass(expected_model_class, models.ft_jnf.FT_JNF):
                model = expected_model_class(window_device=self.test_device)
            else:
                model = expected_model_class()
            actual_model_class = model.__class__

            self.assertEqual(actual_model_class, expected_model_class,
                             msg=f'Object of class {expected_model_class} could not be created!')


    def test_model_processing(self):

        random_input = torch.randn(size=self.expected_input_shape)

        for model_class in self.model_classes_to_test:
            #print(model_class.__name__)

            if issubclass(model_class, models.ft_jnf.FT_JNF):
                model = model_class(window_device=self.test_device)
            else:
                model = model_class()
            output = model(random_input)
            actual_output_shape = output.shape

            self.assertEqual(actual_output_shape, self.expected_output_shape,
                             msg=f'Object of class {model_class} could not input of shape {self.expected_input_shape} to expected shape {self.expected_output_shape}, instead outputs shape {actual_output_shape}!')


    def test_model_training(self):
        loss_fn = torch.nn.MSELoss()
        random_input = torch.randn(size=self.expected_input_shape, requires_grad=True)
        random_target = torch.randn(size=self.expected_output_shape, requires_grad=False)

        for model_class in self.model_classes_to_test:
            if issubclass(model_class, models.ft_jnf.FT_JNF):
                model = model_class(window_device=self.test_device)
            else:
                model = model_class()

            optim = torch.optim.Adam(model.parameters(), lr=0.1)

            # save the initialized weights
            old_weights = []
            for param in model.parameters():
                old_weights.append(param.clone())

            # forward process random input
            output = model(random_input)

            # compute loss and backpropagate
            loss = loss_fn(output, random_target)
            loss.backward()
            optim.step()

            # save the updated weights
            new_weights = []
            for param in model.parameters():
                new_weights.append(param.clone())

            for i, param in enumerate(old_weights):
                self.assertTrue((old_weights[i] != new_weights[i]).any(),
                                msg=f'no parameters changed during training a single step with model of class {model_class}!')


if __name__ == '__main__':
    unittest.main()
