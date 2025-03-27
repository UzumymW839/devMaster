"""
Code originally from Menno: https://gitserv00.idmt.fraunhofer.de/hsa/tools/multichannel-data-generator/-/blob/main/scene/free_field_impulse_response.py
"""
import numpy as np
from scipy.fftpack import ifft

speed_of_sound = 343


def sph2cart(sph_coord,mode='degree'):
    """function to convert spherical to cartesian coordinates"""

    if isinstance(sph_coord,list):
        sph_coord = np.array(sph_coord)

    if sph_coord.ndim>1:
        radius = sph_coord[:,0]
        azimuth = sph_coord[:,1]
        elevation = sph_coord[:,2]
    else:
        radius = sph_coord[0]
        azimuth = sph_coord[1]
        elevation = sph_coord[2]

    # convert from degree to radian
    if mode == 'degree':
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)

    # convert to cartesian coordinates
    x = radius * np.cos(elevation) * np.cos(azimuth)
    y = radius * np.cos(elevation) * np.sin(azimuth)
    z = radius * np.sin(elevation)

    if sph_coord.ndim>1:
        cart_coord = np.concatenate((x[:,None],y[:,None],z[:,None]),axis=1)
    else:
        cart_coord = np.array((x,y,z))

    return cart_coord


def nextpow2(x):
    """computes the next power of two above an arbitrary number. Helper function to get a FFT size that is power of two, that FFT works efficent.

    Parameters: x: int
                                any integer number

    Returns:    power of two: int
                                the next power of two above input x.
    """
    return int(np.ceil(np.log2(np.abs(x))))

def free_field_ir(mic_positions,sound_source_position,samplerate=16000):
    """compute impulse response between a sound source and a microphone array in free-field conditions.

    The free-field impulse responses only contain the time delays between the microphones depending on the position of the sound sources.

    Parameters: mic_positions: 2d numpy array or nested list
                                position of microphones in cartesian coordinates. If numpy array, number of rows is equal to number of microphones. If nested list, each list item contains a list with the coordinates of the microphone.

                sound_source_position: 1d numpy array of list
                                position of the sound source in cartesian coordinates.

                samplerate: int, optional
                                samplerate of the resulting impulse response in Hz. Defaults to 16000 Hz

    Returns:    impulse_responses: nd array
                                resulting impulse responses. If the number of microphones is greater than 2, it will return a 2d array, where the number of columns is equal to the number of microphones.
    """

    # compute distance between microphones and sound source
    mic2source = sound_source_position - mic_positions
    mic2source_distances = np.sqrt(np.sum(mic2source**2,axis=-1))

    # compute time delay between microphones and sound source
    time_delay = mic2source_distances / speed_of_sound

    # choose fft length based on maximum delay
    fftlen = 2**nextpow2(2*np.max(time_delay)*samplerate)
    frequencies = np.linspace(0,int(samplerate/2),int(fftlen/2+1))

    # compute impulse responses via transfer function
    impulse_responses = []
    for tau in time_delay:
        tf = np.exp(-1.j*2*np.pi*frequencies*tau)
        tf = np.concatenate((tf,np.conj(tf[-2:0:-1])))
        impulse_responses.append(np.real(ifft(tf)))
    impulse_responses = np.array(impulse_responses).transpose()

    return impulse_responses



# main function to test
if __name__ == "__main__":
    """example to call function free_field_ir"""
    from numpy.matlib import repmat

    # define an array
    array_geometry = [ # x y z
        [0,0,0],
        [0,0.0035,0],
        [0,-0.0035,0],
        [0.006,0,0],
        [-0.006,0,0],
    ]

    samplerate = 16000

    # define a source position
    source_distance = 5
    source_elevation = 65
    source_azimuth = 0
    source_pos = [source_distance, source_azimuth, source_elevation]

    # convert to cartesian coordinates
    source_pos_cart = sph2cart(source_pos)

    # compute impulse responses
    impulse_responses = free_field_ir(array_geometry,source_pos_cart,samplerate)

    import matplotlib.pyplot as plt
    plt.plot(impulse_responses)
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.xlim([-100+impulse_responses.shape[0]//2, 100+impulse_responses.shape[0]//2])
    plt.savefig('free_field_ir_test.svg')

