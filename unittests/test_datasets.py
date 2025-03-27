import unittest
from datasets.ff3_speech_dataset import FF3_SPEECH_DEFAULT_CONFIG, FF3SpeechDataset, SimFF3SpeechDataset, StftFF3SpeechDataset
from datasets.cvde_simulated_dataset import CVDE_SPEECH_DEFAULT_CONFIG, CVDESimDataset


class TestDatasets(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)


    def test_ff3_speech_item(self):

        len_samples = FF3_SPEECH_DEFAULT_CONFIG['example_length_seconds'] * FF3_SPEECH_DEFAULT_CONFIG['desired_samplerate']


        expected_noisy_shape = (2, len_samples)
        expected_target_shape = (1, len_samples)

        for test_idx in [0, 5, 20, 100]:
            ds = FF3SpeechDataset(noise_transform=None)
            actual_noisy, actual_target = ds.__getitem__(index=test_idx)
            actual_noisy_shape = actual_noisy.shape
            actual_target_shape = actual_target.shape

            self.assertEqual(actual_noisy_shape, expected_noisy_shape)
            self.assertEqual(actual_target_shape, expected_target_shape)


    def test_sim_ff3_speech_item(self):

        len_samples = FF3_SPEECH_DEFAULT_CONFIG['example_length_seconds'] * FF3_SPEECH_DEFAULT_CONFIG['desired_samplerate']


        expected_noisy_shape = (2, len_samples)
        expected_target_shape = (1, len_samples)

        for test_idx in [0, 5, 20, 100]:
            ds = SimFF3SpeechDataset(noise_transform=None)
            actual_noisy, actual_target = ds.__getitem__(index=test_idx)
            actual_noisy_shape = actual_noisy.shape
            actual_target_shape = actual_target.shape

            self.assertEqual(actual_noisy_shape, expected_noisy_shape)
            self.assertEqual(actual_target_shape, expected_target_shape)


    def test_stft_ff3_speech_item(self):

        n_fft = 512
        len_samples = FF3_SPEECH_DEFAULT_CONFIG['example_length_seconds'] * FF3_SPEECH_DEFAULT_CONFIG['desired_samplerate']
        num_frames = len_samples // (n_fft//2) + 1

        expected_noisy_shape = (2, n_fft//2+1, num_frames, 2)
        expected_target_shape = (1, n_fft//2+1, num_frames, 2)

        for test_idx in [0, 5, 20, 100]:
            ds = StftFF3SpeechDataset(noise_transform=None, n_fft=n_fft)
            actual_noisy, actual_target = ds.__getitem__(index=test_idx)
            actual_noisy_shape = actual_noisy.shape
            actual_target_shape = actual_target.shape

            self.assertEqual(actual_noisy_shape, expected_noisy_shape)
            self.assertEqual(actual_target_shape, expected_target_shape)


    def test_cvde_sim_speech_item(self):

        len_samples = CVDE_SPEECH_DEFAULT_CONFIG['example_length_seconds'] * CVDE_SPEECH_DEFAULT_CONFIG['desired_samplerate']


        expected_noisy_shape = (2, len_samples)
        expected_target_shape = (1, len_samples)

        for test_idx in [0, 5, 20, 100]:
            ds = CVDESimDataset(noise_transform=None)
            actual_noisy, actual_target = ds.__getitem__(index=test_idx)
            actual_noisy_shape = actual_noisy.shape
            actual_target_shape = actual_target.shape

            self.assertEqual(actual_noisy_shape, expected_noisy_shape)
            self.assertEqual(actual_target_shape, expected_target_shape)


if __name__ == '__main__':
    unittest.main()
