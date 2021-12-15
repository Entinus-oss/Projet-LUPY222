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
    return v

def main():
    
    num = 50
    t = np.linspace(0, 1, num)
    step = 0.01
    
    data = np.empty((num, num, 2, num))
    print("data shape", data.shape)
    #print("data", data)

    for x in range(-num, 0):
        for v in range(-num, 0):

            initial_values = [x/num, v/num]
            data[x][v] = RK4(derivee, initial_values, step, t)
        
            plt.plot(data[x, v, 0], data[x, v, 1], 'k,', markersize=1) #trajectoire
            plt.plot(initial_values[0], initial_values[1], 'r.', markersize=2) #conditions initiales
            plt.plot(data[x, v, 0, -1], data[x, v, 1, -1], 'b.', markersize=4) #points finals

            #print("#####", data[i][j])

    print("data", data[:, :, 0])

    
    plt.xlabel("déplacement (cm)")
    plt.ylabel("vitesse (cm/s)")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()

