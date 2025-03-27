import pathlib
import tqdm
import numpy as np
import soundfile as sf
import scipy.signal as sig

np.random.seed(456)

def apply_random_reverb(audio_data, rir_filename, output_fs=16000, reference_channel_list=[0]):

    # prepare output data and add empty channels for clean reference signals
    num_input_ch = audio_data.shape[1]
    num_ref_ch = len(reference_channel_list)
    output_data = np.zeros((audio_data.shape[0], num_input_ch+num_ref_ch))

    # read room impulse response (RIR/rir)
    rir, fs_rir = sf.read(file=rir_filename, always_2d=True)
    rir = rir[:, [0]] # if RIR is multi-channel, only use the first
    assert fs_rir == output_fs

    # normalize RIR
    rir /= np.amax(np.abs(rir))

    # compute delay induced by convolving with RIR
    delay_samples = int(np.argmax(np.abs(rir), axis=0))

    # filter audio signal with the RIR
    reverb_data = sig.fftconvolve(in1=audio_data, in2=rir, mode='full', axes=0)
    output_data[:, :num_input_ch] = reverb_data[:output_data.shape[0], :]

    # set up clean reference channels that match the delay induced by RIR filtering
    for ref_ch in reference_channel_list:

        # pad audio signal with zeros corresponding to RIR delay
        reference_channel_data = np.pad(audio_data[:, ref_ch], pad_width=(delay_samples, 0), mode='constant')

        # remove trailing part of clean signal
        output_data[:, num_input_ch+ref_ch] = reference_channel_data[:-delay_samples]

    # add clean delayed signal as extra channel
    return output_data


def main():
    print("generating binaural hearpiece data with reverb...")

    output_fs = 16000

    # path to read multi-channel anechoic speech files from
    anechoic_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/clean_multichannel_speech_array')

    reverb_irs_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/impulse_responses')

    # path to write multi-channel speech files into
    output_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/reverberated_multichannel_speech_array')
    output_path.mkdir(exist_ok=True)

    # get a list all files
    filename_list = list(anechoic_path.iterdir())
    print(f'generating speech data with {len(filename_list)} files ...')

    # set up RIR filename list
    rir_filename_list = [fname for fname in reverb_irs_path.glob('**/*.wav')]
    print(f'using {len(rir_filename_list)} different room impulse responses from SLR26 and SLR28 ...')


    # for each file, use impulse responses to create multi-channel spatial recordings
    for filename in tqdm.tqdm(filename_list):

        # read source file speech signal and resample it
        audio_data, samplerate = sf.read(filename, always_2d=True)
        assert samplerate == output_fs

        # add reverb to the direct speech signal by using a random room impulse response
        rir_idx = np.random.randint(low=0, high=len(rir_filename_list))
        simulated_data = apply_random_reverb(audio_data, rir_filename=rir_filename_list[rir_idx])

        # scale audio to maximum amplitude of 1 (maximum over all channels, level differences are kept)
        simulated_data /= np.amax(np.abs(simulated_data))

        # save the simulated multi-channel speech file
        out_filename = output_path.joinpath(filename.with_suffix('.wav').name)
        sf.write(file=out_filename, data=simulated_data, samplerate=output_fs)


def main_diamond():
    print("generating diamond data with reverb...")

    output_fs = 16000

    # path to read multi-channel anechoic speech files from
    anechoic_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/clean_multichannel_speech_array')

    reverb_irs_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/impulse_responses')

    # path to write multi-channel speech files into
    output_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/reverberated_multichannel_speech_array/')
    output_path.mkdir(exist_ok=True)

    # get a list all files
    filename_list = list(anechoic_path.iterdir())
    print(f'generating speech data with {len(filename_list)} files ...')

    # set up RIR filename list
    rir_filename_list = [fname for fname in reverb_irs_path.glob('**/*.wav')]
    print(f'using {len(rir_filename_list)} different room impulse responses from SLR26 and SLR28 ...')


    # for each file, use impulse responses to create multi-channel spatial recordings
    for filename in tqdm.tqdm(filename_list):

        # read source file speech signal and resample it
        audio_data, samplerate = sf.read(filename, always_2d=True)
        assert samplerate == output_fs

        # add reverb to the direct speech signal by using a random room impulse response
        rir_idx = np.random.randint(low=0, high=len(rir_filename_list))
        simulated_data = apply_random_reverb(audio_data, rir_filename=rir_filename_list[rir_idx])

        # scale audio to maximum amplitude of 1 (maximum over all channels, level differences are kept)
        simulated_data /= np.amax(np.abs(simulated_data))

        # save the simulated multi-channel speech file
        out_filename = output_path.joinpath(filename.with_suffix('.wav').name)
        sf.write(file=out_filename, data=simulated_data, samplerate=output_fs)


if __name__ == '__main__':
    #main()
    main_diamond()
