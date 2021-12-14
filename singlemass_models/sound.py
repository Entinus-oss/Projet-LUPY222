import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read

alpha = 0.32
beta = 100
gamma = 0.78
delta = 0.97

def derivee(u, t):
    '''
        Soit u = (u0, u1)
        Équation d'évolution du pendule : d(u0, u1)/dt = (u1, -k ** 2 * u0)
    '''
    # Initialisation de la dérivée
    du = np.empty(u.shape)
    
    # Dérivée de la vitesse
    du[0] = u[1]
    du[1] = -alpha * (1 + beta * u[0]**2) * u[1] - u[0] + (gamma * u[0])/(1 + u[0] + delta * u[1])

    return du

def RK4(derivee, initial_values, step, t):
    
    """Méthodes de Runge-Kutta d'ordre 4. Renvoie un tableau numpy 
    contenant les valeurs de l'angle theta."""
    
    v = np.empty((2, t.shape[0]))

    # Condition initiale
    v[:, 0] = initial_values 

    # Boucle for
    for i in range(t.size - 1):
        d1 = derivee(v[:, i], t[i])
        d2 = derivee(v[:, i] + d1 * step / 2, t[i] + step / 2)
        d3 = derivee(v[:, i] + d2 * step / 2, t[i] + step / 2)
        d4 = derivee(v[:, i] + d3 * step, t[i] + step)
        v[:, i + 1] = v[:, i] + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)

    # Argument de sortie
    return v[0]

def main():
    
    samplingRate = 44100
    amplitude = 10000

    t = np.arange(0, 1, 1/samplingRate)
    step = 0.01

    initial_values = [0.01, 0.6]

    deplacement = RK4(derivee, initial_values, step, t)
    wavData = amplitude * deplacement

    print(wavData)

    wavFile = write("wav/singlemass_models/test1.wav", samplingRate, wavData.astype("float32"))
    plt.plot(t, wavData)

    plt.xlabel("time (s)")
    plt.ylabel("déplacement du ressort (cm)")
    #plt.ylim(-1, 1)
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()