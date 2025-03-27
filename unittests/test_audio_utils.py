import unittest
import torch

from utils.audio_utils import random_fill_or_cut_speech_audio

class TestAudioUtils(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)


    def test_cut_or_pad(self):

        desired_length_samples = 4*16000

        ch_num = 3
        test_lengths_samples = [5, 22000, 4*16000, 65000, 100000]

        for test_len in test_lengths_samples:

            audio_in = torch.randn(size=(ch_num, test_len))

            audio_out = random_fill_or_cut_speech_audio(audio_in, desired_length_samples)

            actual_length_samples = audio_out.shape[1]
            actual_channels = audio_out.shape[0]

            self.assertEqual(actual_channels, ch_num)
            self.assertEqual(actual_length_samples, desired_length_samples)


if __name__ == '__main__':
    unittest.main()