import numpy as np
import matplotlib.pyplot as plt
import wave
import struct

#Default values are m = 0.15 * 10**(-3) kg, l = 1.4 * 10**(-3) m, T = 3 N
m = 0.15 * 10**(-3) #kg 
l = 1.4 * 10**(-3) #m
T =  3#N

def frequency_from_fundamental(n, round_at=0, logs=True):
    
    """Calculate frequency based on a {mass + ruberband} system."""

    frequency = 1/2 * np.sqrt(n**2*T/(m*l))
    
    if round_at:
        frequency = round(frequency, round_at)
        
    if logs:
        print("fundamental frequency : ", 1/2 * np.sqrt(T/(m*l)), "Hz")
        print("frequency", n,  "is", frequency, "Hz")
        
    return frequency

def add_sine_waves(listOfSineWave):
    
    sumOfSineWaves = np.copy(listOfSineWave[0])

    for i in range(len(listOfSineWave)):
       sumOfSineWaves = np.add(sumOfSineWaves, listOfSineWave[i])

    return sumOfSineWaves

def createWaveFile(listOfSineWave, samplingRate = 44100):

    F = wave.open('test.wav', 'wb')
    F.setnchannels(1)
    F.setsampwidth(2)
    F.setframerate(samplingRate)
    if listOfSineWave.shape[0] > 1:
        for sine in listOfSineWave:
            #print(sine)
            for w in sine:
                #print(type(int(w)))    
                F.writeframes(struct.pack('f', w))
    else:
        for w in listOfSineWave:
            #print(type(int(w)))    
            F.writeframes(struct.pack('f', w))
    F.close()

     
def main():
    num = 100000
    num_harmonics = 3
    amplitude = 10000
    t = np.linspace(0, num, num)
    sine_waves = np.zeros((num_harmonics, num))

    for i in range(sine_waves.shape[0]):
        #create sine wave from frequency
        sine_waves[i] = amplitude * np.sin(2*np.pi*frequency_from_fundamental(i+1, round_at=0) * t)
        #plt.plot(t, sine_waves[i])

    createdSineWave = add_sine_waves(sine_waves)
    #print(createdSineWave)
    createWaveFile(sine_waves, samplingRate=44100)

    plt.plot(t, createdSineWave)

    #plt.xlim(0, 0.1)
    plt.show()

if __name__ == "__main__":
    main()