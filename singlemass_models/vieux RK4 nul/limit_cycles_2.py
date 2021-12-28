import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

alpha = 0.32
beta = 100
gamma = 0.6
delta = 0.97

def derivee(u, t, param):
    '''
        Soit u = (u0, u1)
        Équation d'évolution du pendule : d(u0, u1)/dt = (u1, -k ** 2 * u0)
    '''
    # Initialisation de la dérivée
    du = np.empty(u.shape)
    
    # Dérivée de la vitesse
    du[0] = u[1]
    du[1] = -param[0] * (1 + param[1] * u[0]**2) * u[1] - u[0] + (param[3] * u[1])/(1 + u[0] + param[2] * u[1])

    return du

def RK4(derivee, initial_values, param, step, t):
    
    """Méthodes de Runge-Kutta d'ordre 4. Renvoie un tableau numpy 
    contenant les valeurs de l'angle theta."""
    
    v = np.empty((2, t.shape[0]))

    # Condition initiale
    v[:, 0] = initial_values 

    # Boucle for
    for i in range(t.size - 1):
        d1 = derivee(v[:, i], t[i], param)
        d2 = derivee(v[:, i] + d1 * step / 2, t[i] + step / 2, param)
        d3 = derivee(v[:, i] + d2 * step / 2, t[i] + step / 2, param)
        d4 = derivee(v[:, i] + d3 * step, t[i] + step, param)
        v[:, i + 1] = v[:, i] + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)

    # Argument de sortie
    return v

def main():

    alpha = 0.32
    beta = 100
    gamma = 0
    delta = 0.97

    param = np.array([alpha, beta, delta, gamma])

    nb_iter = 40
    gamma_values = np.round(np.linspace(0, 1, nb_iter), 2)

    step = 1  

    start = 0
    end = 40
    t = np.arange(start, end, step)
    num = t.size

    data = np.empty((num, num, 2, num))
    print("data shape", data.shape)
    #print("data", data)

    initial_values = np.linspace(-1, 1, num)

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')
    
    plot_counter = 0
    total_plot = nb_iter * num * num
    
    for k in range(nb_iter):

        param[3] = gamma_values[k]
        
        for i in range(0, num):

            u = initial_values[i]
            for j in range(0, num):
                
                v = initial_values[j]
                #print(u, v, param[3])

                data[i][j] = RK4(derivee, [u, v], param, step, t)

                #plt.plot(u, v, 'r.', markersize=4) #initials values
                #plt.plot(data[i, j, 0], data[i, j, 1], 'k.', markersize=1) #trajectoire
                ax.plot3D(param[3], data[i, j, 0, -1], data[i, j, 1, -1], 'b.') #points finals
                plot_counter += 1
                print("plotted :", plot_counter, '/', total_plot, end='\r')
    
    print("plotting ...")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$u$")
    ax.set_zlabel(r"$v$")

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.show()

    return 0

if __name__ == "__main__":
    main()

