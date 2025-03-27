import numpy as np
from numba import njit

def active_speech_level(speechData, fs, margin = 15.9, scaleFactor = 1, refValue = 1):
    """ activeSpeechLevel_ITU_P56 - active speech level measurement recommended by ITU P56
        purpose: for information see ITU P56

        Args:
            speechData: input data to measure the speech level
            fs:         sample rate in Hz
            margin:     Margin in dB, difference between threshold and active
                        speech level. Default value is 15.9
            scaleFactor:(Volts/unit) scale factor of the analogue-digital converter;
                        Default is 1 -> no scaling
            refValue:   varaible r in ITU P56, referenz voltage
                        Default 1

        Returns:
            (tuple): tuple containing:
                - speechLevel: level of active speech
                - activityFactor: Speech activity in percent
                - vadInfo: Vad signal, designed to fulfill the activity factor
                           calculated by ITU P56

    # ohlems: removed option to get VAD info for more than one channel, this assumes 1D data!
        also modified this heavily in terms of naming and speedup with njit decorators and so on, might not
        work 100% as before or in the norm, but it is intended for energy-based VAD cutting of clean speech recordings.

        """
    assert speechData.ndim == 1
    sigLen = speechData.shape[0]

    ## initialisation
    t = 1/fs           #sampling period
    T = 0.03           # effective smoothing time in seconds
    g = np.exp(-t/T)     # smoothing coefficient
    H = 0.2            #Hangover time in seconds
    I = np.ceil(H/t)     #Rounded up to next integer
    c_step = 1 #0.5 # smaller steps -> more precision, more computations
    c = np.amax(np.abs(speechData)) * 1./(2**np.arange(16, 0.5, -c_step) )  #treshold vector

    refLevel = 20*np.log10(refValue)
    scaleLevel = 20*np.log10(scaleFactor)

    # Active Speech Level Estimation
    sq = np.sum(np.square(speechData), axis=0)
    qVec, a = compute_q_vec(speechData, sigLen, I, g, c)

    
    A = 10 * np.log10(sq / (a + 2e-16)) + scaleLevel - refLevel
    C = 20 * np.log10(c) + scaleLevel - refLevel

    tmp = A-C
    idx = np.argwhere( tmp<margin )
    idx = idx[0]

    # linear interpolation
    m = tmp[idx] - tmp[idx-1]
    b = tmp[idx] - m*idx
    idxA = (margin-b)/m

    speechLevel = ( A[idx]-A[idx-1] ) * (idxA-idx+1) + A[idx-1]
    activityFactor =( ( a[idx] - a[idx-1] ) * ( idxA-idx+1 ) + a[idx-1]) / sigLen

    # actual VAD
    vadDelay = int(np.round(T*fs))
    vadInfo = compute_vad_info(sigLen, activityFactor, qVec, vadDelay)

    return speechLevel, activityFactor, vadInfo


@njit
def compute_q_vec(speechData, sigLen, I, g, c):

    # initialize running variables and vectors
    p = 0
    q = 0
    hangover_count = 0
    active_count = np.zeros( (len(c),) )
    qVec = np.zeros((sigLen,))

    for n in  range(sigLen):
        p  = g * p + (1-g)*np.abs(speechData[n])
        q  = g * q + (1-g)*np.abs(p)

        qVec[n] = q

        active_count, hangover_count = c_update(c,q,active_count,I,hangover_count)

    return qVec, active_count

@njit
def c_update(c:np.ndarray, q: np.float64, active_count: int, I: int, hangover_count: int):

    for j, c_val in enumerate(c):

        # check thresholds
        if q >= c_val:
            active_count[j] += 1
            hangover_count = 0

        # check hangovers
        elif hangover_count < I:
            active_count[j] += 1
            hangover_count += 1

    return active_count, hangover_count


#@njit
def compute_vad_info(sigLen, activityFactor, qVec, vadDelay, max_comparisons=100):
    vadInfo     = np.bool_( (sigLen,) )

    upperBound  = np.amax(qVec)
    lowerBound = 0
    tmpAF      = 0

    activityFactor_compare = (activityFactor*1000) // 1

    comparision_cnt = 0

    while activityFactor_compare != np.round(tmpAF*1000):
        tmpLim = (upperBound +lowerBound) / 2
        vadInfo   = (tmpLim <= qVec)
        tmpAF  = np.mean(vadInfo)

        if tmpAF > activityFactor:
            lowerBound = tmpLim
        else:
            upperBound = tmpLim

        comparision_cnt += 1

        # after a sufficient amount of comparisons, just stop!
        if comparision_cnt >= max_comparisons:
            break

    # compensate for VAD delay
    vadInfo[0:-vadDelay] = vadInfo[(vadDelay-1):-1]

    return vadInfo


if __name__ == '__main__':
    import pathlib
    import soundfile as sf
    #import matplotlib.pyplot as plt
    import time

    #import cProfile
    #pr =cProfile.Profile()

    fname = pathlib.Path('unittests/example_files/common_voice_de_27023380.wav')
    x, fs = sf.read(fname)
    #print(x.shape) # time, 2ch)

    #print(f'file has length {x.shape[0]/fs} seconds')

    #speech_data = x[:,[0]] / np.amax(np.abs(x[:,[0]])) # normalization doesn't make a difference (file is probably already normalized)

    #t0 = time.time()
    #pr.enable()
    speech_level, activity_factor, vad_info = active_speech_level(speechData=x[:,0], fs=np.float64(fs))
    #pr.disable()
    #pr.print_stats(sort='tottime')
    #print(f'{time.time()-t0}')
    #act_speech_data = x[np.bool_(vad_info.flatten()), :]

    #plt.plot(x[:,0])
    #plt.plot(vad_info)
    #plt.plot(act_speech_data)
    #plt.savefig('vad_test.png')
