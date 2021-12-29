import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read

M, B, K = 0.476, 100, 200000 #Mass g/cm^2 Damping dyne s/cm^3 Stiffness dyne/cm^3
x0, eta = 1 * 10 ** -1, 10000 #cm, /cm/cm phenomenological nonlinear coefficient
Pl = 5000 #dyne/cm^2
T = 0.3 #cm glottal height
c = 100 #cm/s wave velocity
tau = T / (2 * c)

alpha = B / np.sqrt(M * K)
beta = x0 ** 2 * eta
gamma = 2 * tau * Pl / (x0 * np.sqrt(M * K)) # ++ frequence
delta = tau * np.sqrt(K / M)

print(alpha, beta, gamma, delta)

def derivee(u, t):
    '''
        Soit u = (u0, u1)
        Équation d'évolution du pendule : d(u0, u1)/dt = (u1, -k ** 2 * u0)
    '''
    # Initialisation de la dérivée
    du = np.empty(u.shape)
    
    # Dérivée de la vitesse
    du[0] = u[1]
    du[1] = -alpha * (1 + beta * u[0]**2) * u[1] - u[0] + (gamma * u[1])/(1 + u[0] + delta * u[1])

    return du

def RK4(start, end, step, v_ini, derivee, ordre):
    '''
        Application de la méthode rk4
    '''
    # Création du tableau temps
    interval = end - start                     # Intervalle
    num_points = int(interval / step) + 1      # Nombre d'éléments
    t = np.linspace(start, end, num_points)    # Tableau temps t

    # Initialisation du tableau v
    v = np.empty((ordre, num_points))

    # Condition initiale
    v[:, 0] = v_ini 

    # Boucle for
    for i in range(num_points - 1):
        d1 = derivee(v[:, i], t[i])
        d2 = derivee(v[:, i] + step / 2 * d1, t[i] + step / 2)
        d3 = derivee(v[:, i] + step / 2 * d2, t[i] + step / 2)
        d4 = derivee(v[:, i] + step * d3, t[i] + step)
        v[:, i + 1] = v[:, i] + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)

    # Sorties
    return t, v

def main():

    samplingRate = 8000
    amplitude = 10000

    step = 0.1
    start = 0
    end = 500


    initial_values = [0.1, 0]
    
    t, v = RK4(start, end, step, initial_values, derivee, 2)

    sound_data = amplitude * v[0]
    wavFile = write("wav/singlemass_models/nonlinear_viscous_initial_pressure/test.wav", samplingRate, sound_data.astype("float32"))
    
    plt.plot(t, v[0])
    
    plt.xlabel("time (s)")
    plt.ylabel("déplacement du ressort (cm)")
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()