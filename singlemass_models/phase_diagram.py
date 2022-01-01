import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
    
    step = 1

    start = 0
    end = 40

    num = 1000
    initial_values = np.linspace(-1, 1, num)

    plot_counter=0

    for i in range(initial_values.shape[0]):

        u = initial_values[i]

        for j in range(initial_values.shape[0]):

            v = initial_values[j]

            t, v = RK4(start, end, step, [u, v], derivee, 2)

            #plt.plot(u, v, 'r.', markersize=4) #initials values
            plt.plot(v[0], v[1], 'k.', markersize=1) #trajectoire
            plt.plot(v[0, -1], v[1, -1], 'b.', markersize=4) #points finals

            plot_counter += 1
            print("Plotted :", plot_counter, "/", initial_values.shape[0] ** 2, end='\r')


    plt.xlabel("u")
    plt.ylabel("v")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.show()

    return 0

if __name__ == "__main__":
    main()

