import pathlib
import numpy as np
import soundfile as sf
import scipy.signal as sig
import multiprocessing
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.free_field_impulse_response import free_field_ir, sph2cart
import warnings
warnings.filterwarnings("error")

from utils.vad import active_speech_level

np.random.seed(123)

import os
MAX_WORKERS = os.cpu_count()/2

import json


def simulate_point_source(audio_data, ir_matrix, direction_idx):

    # select random impulse response
    random_ir = ir_matrix[direction_idx, :, :] # out: (time, channels)

    simulated_data = sig.fftconvolve(in1=audio_data, in2=random_ir, axes=0)

    return simulated_data


def setup_diamond_irs(output_fs, n_fft):

 # define an array
    # THESE ARE NOT THE CORRECT CHANNEL POSITIONS
    #array_geometry = [ # x y z
    #    [0,0,0],
    #    [0,0.0035,0],
    #    [0,-0.0035,0],
    #    [0.006,0,0],
    #    [-0.006,0,0],
    #]
    # THESE ARE THE CORRECT CHANNEL POSITIONS
    array_geometry = [ # x y z
        [-0.0035,0,0],
        [0,0,0],
        [0,-0.006,0],
        [0.0035,0,0],
        [0,0.006,0]
    ]
    #valid_azi_directions = [22.5, 0, -22.5]
    valid_azi_directions = [30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30]
    for idx in range(len(valid_azi_directions)):
        valid_azi_directions[idx] += 90
    #valid_azi_directions = [0]
    #valid_ele_directions = [0, 35]
    valid_ele_directions = [-10, -5, 0, 5, 10]

    num_samples = n_fft
    num_directions = len(valid_azi_directions) * len(valid_ele_directions)

    ir_matrix = np.zeros(
        (num_directions, num_samples, len(array_geometry))
    )


    # define a source position
    source_distance = 5

    e_offset = len(valid_azi_directions)

    for e_idx, source_elevation in enumerate(valid_ele_directions):
        for d_idx, source_azimuth in enumerate(valid_azi_directions):

            source_pos = [source_distance, source_azimuth, source_elevation]
            
            # convert to cartesian coordinates
            source_pos_cart = sph2cart(source_pos)

            # compute impulse responses
            impulse_responses = free_field_ir(array_geometry,source_pos_cart, output_fs)

            # cut impulse responses down to filter length n_fft
            ir_mic_cut = impulse_responses[:n_fft, :]

            ir_matrix[e_idx*e_offset+d_idx, :, :] = ir_mic_cut

    return ir_matrix


def multi_main_loop(filename_list, output_path, output_fs, ir_matrix):
    print(f'using {len(filename_list)} speech recordings')

    njobs = int(MAX_WORKERS)
    q = multiprocessing.Queue()
    for wavfile in filename_list:
        q.put(wavfile)
    for _ in range(njobs):
        q.put(None)

    pool = []
    for _ in range(njobs):
        p = multiprocessing.Process(target=process_file, args=(q, output_path, output_fs, ir_matrix), daemon=True)
        p.start()
        pool.append(p)
    for p in pool:
        p.join()

def main():

    output_fs = 16000
    n_fft = 512

    # amount of CVDE to be used
    dataset_fraction = 0.05

    # set up impulse response measurements to filter with
    ir_matrix = setup_diamond_irs(
                        output_fs=output_fs,
                        n_fft=n_fft)

    print(f'set up impulse response matrix with shape: {ir_matrix.shape}')

    # path of commonvoice/de source files (single-channel)
    cvde_source_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/cv-corpus-11.0-2022-09-21/de/clips')

    # path to write multi-channel speech files into
    output_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/clean_multichannel_speech_array')
    output_path.mkdir(exist_ok=True)

    # get a list all files
    filename_list = list(cvde_source_path.iterdir())
    file_count = len(filename_list)

    # select a subset of them
    partial_file_count = int(dataset_fraction * file_count)
    filename_list = filename_list[:partial_file_count]

    print(f'CVDE source dataset has {file_count} files, selecting a fraction of {dataset_fraction} ...')
    print(f'generating speech data with {len(filename_list)} files ...')


    # for each file, use impulse responses to create multi-channel spatial recordings
    multi_main_loop(filename_list, output_path, output_fs, ir_matrix)


def process_file(q: multiprocessing.Queue, output_path: pathlib.Path,
                 output_fs: int, ir_matrix: np.ndarray):

    for filename in iter(q.get, None):

        #print(filename)

        # read source file speech signal and resample it
        audio_data, samplerate = sf.read(filename, always_2d=True)
        audio_data = sig.resample_poly(up=output_fs, down=samplerate, x=audio_data, axis=0)

        # randomly select a direction index
        direction_idx = np.random.randint(low=0, high=ir_matrix.shape[0])

        #print(f'direction idx: {direction_idx}')

        # only use the part of the signal where speech is active
        try:
            speech_level, activity_factor, vad_info = active_speech_level(speechData=audio_data[:, 0], fs=np.float64(output_fs))
            audio_data = audio_data[vad_info, :]
        except RuntimeWarning as e:
            warnings.warn(f'VAD failed for file {filename}, skipping ...')
            breakpoint()

        # use the random indices to select impulse responses and use them to filter the source signal
        simulated_data = simulate_point_source(audio_data, ir_matrix,
                                            direction_idx=direction_idx)

        # scale audio to maximum amplitude of 1 (maximum over all channels, level differences are kept)
        simulated_data /= np.amax(np.abs(simulated_data))

        # save the simulated multi-channel speech file
        out_filename = output_path.joinpath(filename.with_suffix('.wav').name)
        sf.write(file=out_filename, data=simulated_data, samplerate=output_fs)


if __name__ == '__main__':
    main()
