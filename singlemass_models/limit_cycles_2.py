import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

    global gamma

    nb_iter = 20
    gamma_values = np.round(np.linspace(0, 1, nb_iter), 2)

    step = 1  

    start = 0
    end = 20

    num = 20 # nombre de points de départs

    initial_values = np.linspace(-1, 1, num)

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    
    plot_counter = 0

    for k in range(nb_iter):

        gamma = gamma_values[k]
        
        for i in range(initial_values.shape[0]):

            u0 = initial_values[i]

            for j in range(initial_values.shape[0]):
                
                v0 = initial_values[j]
                #print(u0, v0)

                t, v = RK4(start, end, step, [u0, v0], derivee, 2)

                #plt.plot(u, v, 'r.', markersize=4) #initials values
                #plt.plot(data[i, j, 0], data[i, j, 1], 'k.', markersize=1) #trajectoire
                ax.plot3D(gamma_values[k], v[0, -1], v[1, -1], 'b.') #points finals

                plot_counter += 1
                print("plotted :", plot_counter, '/', initial_values.shape[0] ** 2 * nb_iter, end='\r')
    
    print("plotting ...")

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$u$")
    ax.set_zlabel(r"$v$")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

    return 0

if __name__ == "__main__":
    main()

