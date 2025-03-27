import pathlib
import tqdm
import soundfile as sf
import scipy.signal as sig
import numpy as np

np.random.seed(0)

FS_OUT = 16000
NFFT = 4096


SELECT_MEAS_CH = [3, 5] # OEM-L, OEM-R

N_DIRECTIONS = 8

TALKER_LIST = [
    'VP_01', 'VP_02', 'VP_04', 'VP_05', 'VP_06', 'VP_10', 'VP_11', 'VP_13', 'VP_14',
    'VP_15', 'VP_18', 'VP_21', 'VP_23', 'VP_24', 'VP_26', 'VP_27', 'VP_28', 'VP_29']

def simulate_point_source(audio_data, ir_matrix, vp_idx):

    # select random impulse response
    rnd_direction_idx = np.random.randint(low=0, high=N_DIRECTIONS)
    random_ir = ir_matrix[vp_idx, rnd_direction_idx, :, :]

    #print(rnd_direction_idx)

    filtered_iem_noise = sig.lfilter(b=random_ir[:, 0], a=1, x=audio_data, axis=0)
    filtered_oem_noise = sig.lfilter(b=random_ir[:, 1], a=1, x=audio_data, axis=0)

    simulated_data = np.stack([filtered_iem_noise, filtered_oem_noise], axis=1)

    return simulated_data


def simulate_diffuse(audio_data, ir_matrix, vp_idx, offset_sec=1):
    shift_samples = offset_sec * FS_OUT
    simulated_data = np.zeros((audio_data.shape[0], 2))

    for direction_idx in range(N_DIRECTIONS):
        directional_ir = ir_matrix[vp_idx, direction_idx, :]


        audio_data = np.roll(audio_data, shift=shift_samples, axis=0)

        filtered_iem_noise = sig.lfilter(b=directional_ir[:, 0], a=1, x=audio_data, axis=0)
        filtered_oem_noise = sig.lfilter(b=directional_ir[:, 1], a=1, x=audio_data, axis=0)

        simulated_direction_data = np.stack([filtered_iem_noise, filtered_oem_noise], axis=1)

        simulated_data += (1 / N_DIRECTIONS) * simulated_direction_data

    #print(f'diffuse directions: {direction_idx}')

    return simulated_data


def np_rms(x):
    return np.sqrt(np.mean(np.square(x), axis=0))


def setup_irs(ir_path):

    num_talkers = len(TALKER_LIST)
    num_samples = NFFT

    ir_matrix = np.zeros(
        (num_talkers, N_DIRECTIONS, num_samples, 2)
    )

    for t_idx, talker in enumerate(TALKER_LIST):
        for a_idx in range(N_DIRECTIONS):
            file_str = f'data_{a_idx}.npz'
            ir_fname = ir_path.joinpath(talker, file_str)
            meas_data = np.load(ir_fname)
            fs = meas_data['samplerate']

            ir_mic = meas_data['ir_mic']
            ir_mic = sig.resample_poly(up=FS_OUT, down=fs, x=ir_mic, axis=0)

            ir_mic_cut = ir_mic[:NFFT, SELECT_MEAS_CH]

            ir_matrix[t_idx, a_idx, :, :] = ir_mic_cut

    return ir_matrix


def main(path_noise, output_folder, ir_path, time_secs=10):

    ir_matrix = setup_irs(ir_path)

    for talker_str in TALKER_LIST:
        output_folder.joinpath(talker_str).mkdir(exist_ok=True)

    file_list = [fname for fname in path_noise.iterdir()]

    for f_idx, fname in enumerate(tqdm.tqdm(file_list)):

        audio_data, fs  = sf.read(fname, always_2d=True)
        audio_data = sig.resample_poly(up=FS_OUT, down=fs, x=audio_data, axis=0)

        audio_data = audio_data[:, 0]

        if audio_data.shape[0] < time_secs * FS_OUT:
            print(f'skipping file: {fname}')
            continue

        # randomly choose point source or diffuse noise
        field_idx = np.random.randint(low=0, high=2)
        #print(f'field idx: {field_idx}')

        # randomly choose talker index
        vp_idx = np.random.randint(low=0, high=len(TALKER_LIST))
        #print(f'vp_idx: {vp_idx}')

        # simulate noise field
        if field_idx == 0:
            # point source
            simulated_data = simulate_point_source(audio_data, ir_matrix, vp_idx)

        elif field_idx == 1:
            # diffuse
            simulated_data = simulate_diffuse(audio_data, ir_matrix, vp_idx)

        else:
            raise ValueError(f'field idx {field_idx} is not valid!')

        # scale audio to maximum amplitude of 1
        simulated_data /= np.amax(np.abs(simulated_data))

        out_fname = output_folder.joinpath(TALKER_LIST[vp_idx], fname.name)
        #print(out_fname)

        sf.write(file=out_fname, data=simulated_data, samplerate=FS_OUT)


if __name__ == '__main__':

    path_noise = pathlib.Path(r'C:\Users\ohlems\PhD\code\hearpiece_sim_piero\noise_fullband')
    output_folder = pathlib.Path(r'C:\Users\ohlems\PhD\code\hearpiece_sim_piero\personalized_noise\multichannel_noise')
    ir_path = pathlib.Path(r'C:\Users\ohlems\Seafile\transfer_function_measurements_ff3')

    output_folder.mkdir(exist_ok=True)

    main(path_noise=path_noise,
         output_folder=output_folder,
         ir_path=ir_path)
