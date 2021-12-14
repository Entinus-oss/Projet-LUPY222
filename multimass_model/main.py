import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
from scipy.io.wavfile import write, read

#Default values are m = 0.15 * 10**(-3) kg, l = 1.4 * 10**(-3) m, T = 3 N
m = 4.6 * 10**(-3) #kg 
l = 1.4 * 10**(-3) #m
T =  1#N

def frequencyFromFundamental(n, round_at=0, logs=False):
    
    """Calculate frequency based on a {mass + ruberband} system."""

    frequency = 1/np.pi * np.sqrt(n**2*T/(m*l))
    
    if round_at:
        frequency = round(frequency, round_at)
        
    if logs:
        print("fundamental frequency : ", 1/np.pi * np.sqrt(T/(m*l)), "Hz")
        print("frequency", n,  "is", frequency, "Hz")
        
    return frequency

def addSineWaves(listOfSineWave, normalize=True, logs=False):
    
    sumOfSineWaves = np.copy(listOfSineWave[0])

    if logs: 
        print("sumOfSineWaves", sumOfSineWaves)
        print(listOfSineWave.ndim)

    for i in range(1, len(listOfSineWave)):
       sumOfSineWaves += listOfSineWave[i]
       #print("sum at", i, ",", sumOfSineWaves)

    if logs:
        print("sumOfSineWaves :", sumOfSineWaves)

    if normalize:
        sumOfSineWaves *= 1/np.max(sumOfSineWaves)

    return sumOfSineWaves


    F = wave.open(str(name) + '.wav', 'wb')
    F.setnchannels(1)
    F.setsampwidth(2)
    F.setframerate(samplingRate)

    if logs:
        print("dimension of listOfSines array :", listOfSineWave.ndim)

    if listOfSineWave.ndim == 1:
        for w in listOfSineWave:
            F.writeframes(struct.pack('f', w))
    else:
        for sine in listOfSineWave:
            print(sine.shape)
            for w in sine:
                #print(type(int(w)))    
                F.writeframes(struct.pack('f', w))
    F.close()
    print(str(name) + ".wav successfully created!")

    return str(name) + ".wav"


    """Decompose wav file into int16 array. Return a time linspace to plot and the array"""

    framerate, frames = read(f)

    return framerate, frames

def getSpectrumFromSample(f):

    """Give the spectrum of a wav file"""

    _, sample = read(f)

    return np.real(np.fft.fft(sample))

def main():
    samplingRate = 44100
    samplingInterval = 1/samplingRate

    nLowHarmonics = 9
    nHighHarmonics = 0
    amplitudelow = 0.1
    amplitudehigh = 70000
    initialPhases = 9

    t = np.arange(0, 1, samplingInterval)
    lowFrequencySineWaves = np.zeros((nLowHarmonics, t.size))
    highFrequencySineWaves = np.zeros((nHighHarmonics, t.size))

    frequencies = []

    #Creating the lower harmonics from the lower fundamental
    for i in range(nLowHarmonics):
        #create sine wave from lower frequencies
        frequencies.append(frequencyFromFundamental(i+1, round_at=0))
        lowFrequencySineWaves[i] = np.exp(-amplitudelow *i) * np.cos(2*np.pi * frequencyFromFundamental(i+1, round_at=0, logs=True) * t + (t+2*i*np.pi/initialPhases))
        #plt.plot(t, sine_waves[i])
    #print("lowFrequencySineWaves", lowFrequencySineWaves)
    
    #Creating the higher harmonics from the higher fundamental
    for i in range(nHighHarmonics):
        #create sine wave from lower frequencies
        highFrequencySineWaves[i] = amplitudehigh * np.sin(2*np.pi*frequencyFromFundamental(i+1, round_at=0, logs=False) * t + (t+2*i*np.pi/initialPhases))
        #print("highFrequencySineWaves", highFrequencySineWaves)

    #Group every sine waves into one 2D-array
    sineWaves = np.concatenate((lowFrequencySineWaves, highFrequencySineWaves))

    #Add sine wave together using add method
    createdSineWave = addSineWaves(sineWaves, logs=False)

    wavFile = write("test.wav", samplingRate, createdSineWave.astype('float32')) ### TRES IMPORTANT DE METTRE ASTYPE('FLOAT32')

    #Compute the ouput from sine wave original array
    waveFromIfft = np.fft.ifft2(sineWaves)
    #print(waveFromIfft)


    # plot 
    
    xmax = 0.01
    
    fig, axs = plt.subplots(4)
    axs[0].plot(t, createdSineWave)
    axs[0].set_xlim([0, xmax])

    modelSpectrum = getSpectrumFromSample("test.wav")
    #refSpectrum = getSpectrumFromSample("wav/reference.wav")
    axs[1].semilogx(modelSpectrum)
    #axs[1].semilogx(refSpectrum)
    axs[1].set_ylim(0)
    axs[1].set_ylim([0, 10000])

    for wave in sineWaves:
        axs[2].plot(t, wave)
    axs[2].set_xlim([0, xmax])
    
    #Visualize wav file 
    framerate, frames = read("test.wav")
    t_wav = np.arange(0, len(frames))/framerate
    axs[3].plot(t_wav, frames)
    axs[3].set_xlim([0, xmax])

    plt.show()

    modelSpectrum = getSpectrumFromSample("test.wav")
    #refSpectrum = getSpectrumFromSample("wav/reference.wav")
    #printplt.loglog(refSpectrum,'k')
    plt.loglog(modelSpectrum, 'r')
   

    plt.show()


if __name__ == "__main__":
    main()