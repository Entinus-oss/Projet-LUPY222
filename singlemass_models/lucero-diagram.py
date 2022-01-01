import numpy as np
import matplotlib.pyplot as plt

alpha = 0.52
beta = 100
gamma = 0.78   
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
    
    step = 0.1

    start = 0
    end = 20

    initial_values=np.array([[0.1, 0.1]])

    for i in range(initial_values.shape[0]):
        t, v = RK4(start, end, step, initial_values[i], derivee, 2)
        plt.plot(v[0], v[1], '-', label=str(initial_values[i]))


    plt.plot([-1, 0], [0, -1], 'r--', label="limite")
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

