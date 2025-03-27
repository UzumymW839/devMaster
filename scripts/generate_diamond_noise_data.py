import pathlib
import tqdm
import soundfile as sf
import scipy.signal as sig
import numpy as np
import sys
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.free_field_impulse_response import free_field_ir, sph2cart

np.random.seed(0)

FS_OUT = 16000
NFFT = 512

N_DIRECTIONS = 8


def simulate_point_source(audio_data, ir_matrix):

    # select random impulse response
    rnd_direction_idx = np.random.randint(low=0, high=N_DIRECTIONS)
    random_ir = ir_matrix[rnd_direction_idx, :, :]

    #print(rnd_direction_idx)
    sig_list = []
    for n_ch in range(ir_matrix.shape[-1]):
        filtered_noise = sig.lfilter(b=random_ir[:, n_ch], a=1, x=audio_data, axis=0)
        sig_list.append(filtered_noise)

    simulated_data = np.stack(sig_list, axis=1)

    return simulated_data


def simulate_diffuse(audio_data, ir_matrix, offset_sec=1):
    shift_samples = offset_sec * FS_OUT
    simulated_data = np.zeros((audio_data.shape[0], ir_matrix.shape[-1]))

    for direction_idx in range(N_DIRECTIONS):
        directional_ir = ir_matrix[direction_idx, :]


        audio_data = np.roll(audio_data, shift=shift_samples, axis=0)

        sig_list = []
        for n_ch in range(ir_matrix.shape[-1]):
            filtered_noise = sig.lfilter(b=directional_ir[:, n_ch], a=1, x=audio_data, axis=0)
            sig_list.append(filtered_noise)

        simulated_direction_data = np.stack(sig_list, axis=1)

        simulated_data += (1 / N_DIRECTIONS) * simulated_direction_data

    #print(f'diffuse directions: {direction_idx}')

    return simulated_data


def np_rms(x):
    return np.sqrt(np.mean(np.square(x), axis=0))


def setup_diamond_irs(n_fft):

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
    valid_azi_directions = [22.5 + x*45 for x in range(8)]

    num_samples = n_fft
    num_directions = len(valid_azi_directions)

    ir_matrix = np.zeros(
        (num_directions, num_samples, len(array_geometry))
    )


    # define a source position
    source_distance = 5
    source_elevation = 0


    for d_idx, source_azimuth in enumerate(valid_azi_directions):

        source_pos = [source_distance, source_azimuth, source_elevation]
        # convert to cartesian coordinates
        source_pos_cart = sph2cart(source_pos)

        # compute impulse responses
        impulse_responses = free_field_ir(array_geometry,source_pos_cart, FS_OUT)

        # cut impulse responses down to filter length n_fft
        ir_mic_cut = impulse_responses[:n_fft, :]

        ir_matrix[d_idx, :, :] = ir_mic_cut

    return ir_matrix


def main(path_noise, output_folder, time_secs=10):

    ir_matrix = setup_diamond_irs(n_fft=NFFT)

    file_list = [fname for fname in path_noise.iterdir()]

    for f_idx, fname in enumerate(tqdm.tqdm(file_list)):

        audio_data, fs = sf.read(fname, always_2d=True)
        audio_data = sig.resample_poly(up=FS_OUT, down=fs, x=audio_data, axis=0)

        audio_data = audio_data[:, 0]

        if audio_data.shape[0] < time_secs * FS_OUT:
            print(f'skipping file: {fname}')
            continue

        # randomly choose point source or diffuse noise
        field_idx = np.random.randint(low=0, high=2)
        #print(f'field idx: {field_idx}')

        # simulate noise field
        if field_idx == 0:
            # point source
            simulated_data = simulate_point_source(audio_data, ir_matrix)

        elif field_idx == 1:
            # diffuse
            simulated_data = simulate_diffuse(audio_data, ir_matrix)

        else:
            raise ValueError(f'field idx {field_idx} is not valid!')

        # scale audio to maximum amplitude of 1
        simulated_data /= np.amax(np.abs(simulated_data))

        out_fname = output_folder.joinpath(fname.name)
        #print(out_fname)

        sf.write(file=out_fname, data=simulated_data, samplerate=FS_OUT)


if __name__ == '__main__':
    path_noise = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/noise_fullband')

    output_folder = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/multichannel_noise_array')
    # create output folder if it does not exist
    if not output_folder.exists():
        os.makedirs(output_folder)

    main(path_noise=path_noise,
         output_folder=output_folder)
