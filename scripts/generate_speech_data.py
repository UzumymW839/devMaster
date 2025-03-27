import pathlib
import numpy as np
import soundfile as sf
import scipy.signal as sig
import multiprocessing

from utils.vad import active_speech_level

np.random.seed(123)

import os
MAX_WORKERS = os.cpu_count()/2


def simulate_point_source(audio_data, ir_matrix, subject_idx, direction_idx):

    # select random impulse response
    random_ir = ir_matrix[subject_idx, direction_idx, :, :] # out: (time, channels)

    simulated_data = sig.fftconvolve(in1=audio_data, in2=random_ir, axes=0)

    return simulated_data


def setup_irs(ir_path, talker_list, output_fs, valid_direction_indices,
              select_channels_measurement, n_fft):

    num_talkers = len(talker_list)
    num_samples = n_fft
    num_directions = len(valid_direction_indices)

    ir_matrix = np.zeros(
        (num_talkers, num_directions, num_samples, 2)
    )

    for t_idx, talker in enumerate(talker_list):
        for d_idx in range(num_directions):

            # select valid directions' measurement
            measurement_index = valid_direction_indices[d_idx]
            file_str = f'data_{measurement_index}.npz'

            # read the impulse response file
            ir_fname = ir_path.joinpath(talker, file_str)
            meas_data = np.load(ir_fname)
            fs = meas_data['samplerate']

            ir_mic = meas_data['ir_mic']

            # resample if required
            ir_mic = sig.resample_poly(up=output_fs, down=fs, x=ir_mic, axis=0)

            # cut impulse responses down to filter length n_fft and select desired channels
            ir_mic_cut = ir_mic[:n_fft, select_channels_measurement]

            ir_matrix[t_idx, d_idx, :, :] = ir_mic_cut

    return ir_matrix


def multi_main_loop(filename_list, output_path, output_fs, ir_matrix, talker_list, valid_direction_indices):
    print(f'using {len(filename_list)} speech recordings')

    njobs = int(MAX_WORKERS)
    q = multiprocessing.Queue()
    for wavfile in filename_list:
        q.put(wavfile)
    for _ in range(njobs):
        q.put(None)

    pool = []
    for _ in range(njobs):
        p = multiprocessing.Process(target=process_file, args=(q, output_path, output_fs, ir_matrix, talker_list, valid_direction_indices), daemon=True)
        p.start()
        pool.append(p)
    for p in pool:
        p.join()

def main():

    output_fs = 16000
    n_fft = 4096

    # amount of CVDE to be used
    dataset_fraction = 0.05

    talker_list = [
        'VP_01', 'VP_02', 'VP_04', 'VP_05', 'VP_06', 'VP_10', 'VP_11', 'VP_13', 'VP_14',
        'VP_15', 'VP_18', 'VP_21', 'VP_23', 'VP_24', 'VP_26', 'VP_27', 'VP_28', 'VP_29'
    ]

    #select_channels_measurement=[2,3,4,5] # IEM-L, OEM-L, IEM-R, OEM-R
    select_channels_measurement=[3,5] # OEM-L, OEM-R

    valid_direction_indices = [0, 7] # 22.5° and 360°-22.5° -> semi-frontal directions

    # path of commonvoice/de source files (single-channel)
    cvde_source_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/ohlems/phd/asr_phoneme/commonvoice_de_audio/cv-corpus-11.0-2022-09-21/de/clips')

    ir_measurements_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/ohlems/work/BSE_data/transfer_function_measurements_ff3')

    # path to write multi-channel speech files into
    output_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/ohlems/work/BSE_data/clean_multichannel_speech')
    output_path.mkdir(exist_ok=True)

    # get a list all files
    filename_list = list(cvde_source_path.iterdir())
    file_count = len(filename_list)

    # select a subset of them
    partial_file_count = int(dataset_fraction * file_count)
    filename_list = filename_list[:partial_file_count]

    print(f'CVDE source dataset has {file_count} files, selecting a fraction of {dataset_fraction} ...')
    print(f'generating speech data with {len(filename_list)} files ...')

    # set up impulse response measurements to filter with
    ir_matrix = setup_irs(ir_path=ir_measurements_path,
                        talker_list=talker_list,
                        output_fs=output_fs,
                        valid_direction_indices=valid_direction_indices,
                        select_channels_measurement=select_channels_measurement,
                        n_fft=n_fft)

    print(f'set up impulse response matrix with shape: {ir_matrix.shape}')

    # for each file, use impulse responses to create multi-channel spatial recordings
    multi_main_loop(filename_list, output_path, output_fs, ir_matrix, talker_list, valid_direction_indices)


def process_file(q: multiprocessing.Queue, output_path: pathlib.Path,
                 output_fs: int, ir_matrix: np.ndarray,
                 talker_list: list, valid_direction_indices: list):

    for filename in iter(q.get, None):

        #print(filename)

        # read source file speech signal and resample it
        audio_data, samplerate = sf.read(filename, always_2d=True)
        audio_data = sig.resample_poly(up=output_fs, down=samplerate, x=audio_data, axis=0)

        # randomly select a subject and direction index
        subject_idx = np.random.randint(low=0, high=len(talker_list))
        direction_idx = np.random.randint(low=0, high=len(valid_direction_indices))

        #print(f'subject idx: {subject_idx}')
        #print(f'direction idx: {direction_idx}')

        # only use the part of the signal where speech is active
        speech_level, activity_factor, vad_info = active_speech_level(speechData=audio_data[:, 0], fs=np.float(output_fs))
        audio_data = audio_data[vad_info, :]

        # use the random indices to select impulse responses and use them to filter the source signal
        simulated_data = simulate_point_source(audio_data, ir_matrix,
                                            subject_idx=subject_idx,
                                            direction_idx=direction_idx)

        # scale audio to maximum amplitude of 1 (maximum over all channels, level differences are kept)
        simulated_data /= np.amax(np.abs(simulated_data))

        # save the simulated multi-channel speech file
        out_filename = output_path.joinpath(filename.with_suffix('.wav').name)
        sf.write(file=out_filename, data=simulated_data, samplerate=output_fs)


if __name__ == '__main__':
    main()
