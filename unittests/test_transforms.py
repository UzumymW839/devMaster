import unittest
import torch
import transforms


class TestNoiseTransform(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_snr_fixed(self):

        for target_snr_db in [17, 3, -5]:
            tr = transforms.NoiseTransform(snr_db=target_snr_db)
            actual_snr_db = tr.get_desired_snr_db()

            self.assertEqual(actual_snr_db, target_snr_db)


    def test_snr_range(self):

        snr_ranges_to_test = [
            [-5, 20],
            [6,100],
            [-20, 0]
        ]

        # try each range this many times
        n_samples = 5

        for target_snr_range_db in snr_ranges_to_test:
            for _ in range(n_samples):
                tr = transforms.NoiseTransform(snr_db=target_snr_range_db)
                actual_snr_db = tr.get_desired_snr_db()

                self.assertTrue(target_snr_range_db[0] <= actual_snr_db < target_snr_range_db[1])


class TestNormalizationTransform(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_statistics(self):
        ref_ch = 0
        norm_tf = transforms.Normalization(reference_channel_target=ref_ch)

        noisy_input = torch.randn(2, 4*16000)
        noisy_input[1,:] /= 5

        target_scale = 0.1
        target_offset = 0.01
        target_input = target_scale*torch.randn(1, 4*16000) + target_offset

        desired_noisy_mean = [0,0]
        desired_noisy_std = [1,1]
        desired_target_mean = [target_offset]
        desired_target_std = [target_scale * desired_noisy_std[ref_ch]]

        noisy_output, target_output = norm_tf(noisy_input, target_input)

        noisy_out_mean = torch.mean(noisy_output, dim=1)
        noisy_out_std = torch.std(noisy_output, dim=1)
        target_out_mean = torch.mean(target_output, dim=1)
        target_out_std = torch.std(target_output, dim=1)

        torch.testing.assert_allclose(desired_noisy_mean, noisy_out_mean, atol=1e-2, rtol=1e-2)
        torch.testing.assert_allclose(desired_noisy_std, noisy_out_std, atol=1e-2, rtol=1e-2)
        torch.testing.assert_allclose(desired_target_mean, target_out_mean, atol=1e-2, rtol=1e-2)
        torch.testing.assert_allclose(desired_target_std, target_out_std, atol=1e-2, rtol=1e-2)


    """
    def test_randnoise(self):
        norm_tf = transforms.Normalization(reference_channel_target=0)
        inrand = 5*torch.rand(2, 3*16000) + 1
        inrand[1, :] += 7
        inrand2 = -3 * torch.randn(1, 3*16000) + 5
        in_n, in2 = norm_tf(inrand, inrand)
        print(torch.mean(in_n, dim=1))
        print(torch.std(in_n, dim=1))

        print(torch.mean(in2, dim=1))
        print(torch.std(in2, dim=1))
    """


if __name__ == '__main__':
    unittest.main()
