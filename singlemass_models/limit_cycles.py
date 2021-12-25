import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def derivee(u, t, param):
    '''
        Soit u = (u0, u1)
        Équation d'évolution du pendule : d(u0, u1)/dt = (u1, -k ** 2 * u0)
        param est un np.array contenant alpha beta delta gamma dans cet ordre
    '''
    # Initialisation de la dérivée
    du = np.empty(u.shape)
    #print(alpha, beta, delta, gamma)
    #Dérivée de la vitesse
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

    return v

def solve_edf(initial_values, param, num, step, t):
    
    data = np.empty([num, num, 2, num])

    for i in range(num):

        u = initial_values[i]

        for j in range(num):

            v = initial_values[j]
            #print(u, v)

            data[i][j] = RK4(derivee, [u, v], param, step, t)

            #plt.plot(u, v, 'r.', markersize=4) #initials values
            #plt.plot(data[i, j, 0], data[i, j, 1], 'k.', markersize=1) #trajectoire
            #plt.plot(data[i, j, 0, -1], data[i, j, 1, -1], 'b.', markersize=4) #points finals

    return data

def limit_cycles_g():
    return 100

def main():
    alpha = 0.32
    beta = 100
    gamma = 0.78
    delta = 0.97

    param = np.array([alpha, beta, delta, gamma])

    step = 1

    start = 0
    end = 10
    t = np.arange(start, end, step)
    num = t.size

    initial_values = np.linspace(-1, 1, num)

    nb_iter = 50
    gamma_values = np.linspace(0, 1, nb_iter)

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')

    for k in range(gamma_values.size):
        
        param[3] = round(gamma_values[k], 2)
        new_data = solve_edf(initial_values, param, num, step, t)

        for i in range(num):

            for j in range(num):

                #print(data[k, i, j, 0, -1].shape, data[k, i, j, 0, -1])
                #plt.plot(u, v, 'r.', markersize=4) #initials values
                #plt.plot(data[i, j, 0], data[i, j, 1], 'k.', markersize=1) #trajectoire
                #plt.plot(data[4, i, j, 0, -1], data[4, i, j, 0, -1], 'b.', markersize=4) #points finals
                print(param[3], new_data[i, j, 0, -1], new_data[i, j, 1, -1])
                ax.plot3D(new_data[i, j, 0, -1], new_data[i, j, 1, -1], param[3], 'b.')

    #print(delta_values)
    #   Affichage du résultat

    #ax.plot_surface(X, Y, Z, color = 'y', alpha = 0.3) # Le zorder est incorrect, mais je n'arrive pas à le changer

    plt.xlabel(r"$u$")
    plt.ylabel(r"$v$")
    ax.set_zlabel(r"$\lambda$")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()