import numpy as np
import matplotlib.pyplot as plt

M, B, K = 0.476, 100, 200000 #Mass g/cm^2 Damping dyne s/cm^3 Stiffness dyne/cm^3
x0, eta = 1 * 10 ** -1, 10000 #cm, /cm/cm phenomenological nonlinear coefficient
Pl = 8000 #dyne/cm^2
T = 0.3 #cm glottal height
c = 100 #cm/s wave velocity
tau = T / (2 * c)

alpha = B / np.sqrt(M * K)
beta = x0 ** 2 * eta
gamma = 2 * tau * Pl / (x0 * np.sqrt(M * K)) # ++ frequence
delta = tau * np.sqrt(K / M)

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
    
    t = np.arange(0, 100, 1)

    step = 0.5

    initial_values = [0.1, 0]
    
    deplacement = RK4(derivee, initial_values, step, t)

    plt.plot(t, deplacement)

    plt.xlabel("time (s)")
    plt.ylabel("déplacement du ressort (cm)")
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()
    