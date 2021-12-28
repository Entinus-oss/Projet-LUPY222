#!UTF-8python 
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.32
beta = 1
gamma = 0.4 
delta = 0.97

def derivee(u, t):
    """
        Soit u = (u0, u1)
        Équation d'évolution du pendule : d(u0, u1)/dt = (u1, -k ** 2 * u0)
    """
    # Initialisation de la dérivée
    du = np.empty(u.shape)
    
    # Dérivée de la vitesse
    du[0] = u[1]
    du[1] = -alpha * (1 + beta * u[0]**2) * u[1] - u[0] + (gamma * u[1])/(1 + u[0] + delta * u[1])

    return du

def RK4(derivee, initial_values, step, t, toList=False):
    
    """
        Méthodes de Runge-Kutta d'ordre 4. Renvoie un tableau numpy 
        contenant les valeurs de l'angle theta.
    """
    
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
    if tuple:
        return [v[0], v[1]]
    else: 
        return v

def main():
    
    step = 0.01

    start = 0
    end = 40
    t = np.arange(start, end, step)

    nlines = 7
    data = [0] * nlines
    initial_values = [0] * nlines

    initial_values[0] = [0, 0]
    data[0] = RK4(derivee, initial_values[0], step, t, toList=True)

    initial_values[1] = [-0.03, 0.05]
    data[1] = RK4(derivee, initial_values[1], step, t, toList=True)

    initial_values[2] = [0.047, -0.1]
    data[2] = RK4(derivee, initial_values[2], step, t, toList=True)

    initial_values[3] = [0.049, -0.01]
    data[3] = RK4(derivee, initial_values[3], step, t, toList=True)

    initial_values[4] = [-0.081, 0.05]
    data[4] = RK4(derivee, initial_values[4], step, t, toList=True)

    initial_values[5] = [0.0489, -0.1]
    data[5] = RK4(derivee, initial_values[5], step, t, toList=True)

    initial_values[6] = [0.0499, -0.1]
    data[6] = RK4(derivee, initial_values[6], step, t, toList=True)

    for i in range(len(data)):
        plt.plot(initial_values[i][0], initial_values[i][1], 'r.')
        plt.plot(data[i][0], data[i][1], '-', label=str(initial_values[i]))
        plt.plot(data[i][0][-1], data[i][1][-1], 'b.')

    plt.plot([-1, 0], [0, -1], 'r--', label="limite")
    print(data)
    plt.xlabel("u")
    plt.ylabel("v")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.grid()
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()

