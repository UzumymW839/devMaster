import unittest
import torch

from dataloaders.collate_functions import mix_collate_fn

class CollateFunctionTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_mix_collate(self):

        random_input = torch.randn(2, 16000)
        random_target = torch.randn(1, 16000)

        basic_batch = [(random_input, random_target)]
        actual_batch_size = len(basic_batch)

        collated_inputs, collated_targets = mix_collate_fn(basic_batch)
        output_batch_size = collated_inputs.shape[0]
        self.assertEqual(output_batch_size, actual_batch_size,
                         msg=f'expected batch size {actual_batch_size} but got {output_batch_size}!')

        torch.testing.assert_allclose(random_input, collated_inputs[0], msg='collated input is not the same as the actual input')
        torch.testing.assert_allclose(random_target, collated_targets[0], msg='collated target is not the same as the actual target')


    def test_mix_collate_stft_shaped(self):

        # shape: channels, n_freq, time_frames, real+imag
        random_input = torch.randn(2, 257, 123, 2) # outdated real+imag format
        random_target = torch.randn(1, 257, 123, 2)

        basic_batch = [(random_input, random_target)]
        actual_batch_size = len(basic_batch)

        collated_inputs, collated_targets = mix_collate_fn(basic_batch)
        output_batch_size = collated_inputs.shape[0]
        self.assertEqual(output_batch_size, actual_batch_size,
                         msg=f'expected batch size {actual_batch_size} but got {output_batch_size}!')

        torch.testing.assert_allclose(random_input, collated_inputs[0], msg='collated input is not the same as the actual input')
        torch.testing.assert_allclose(random_target, collated_targets[0], msg='collated target is not the same as the actual target')



if __name__ == '__main__':
    unittest.main()
