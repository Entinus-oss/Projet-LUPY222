import numpy as np
import matplotlib.pyplot as plt


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

def computePhaseData(num, t, step):
    
    """Compute RK4 method numberOfIteration times with different values of
    speed and displacement. t must be a numpy array"""
    
    data = np.empty((num, num, 2, num))
    print("data shape", data.shape)
    #print("data", data)

    for x in range(0, num):
        for v in range(0, num):

            initial_values = [x/num, v/num]
            data[x][v] = RK4(derivee, initial_values, step, t)
    
    return data

def main():
    
    step = 1

    start = 0
    end = 100
    t = np.arange(start, end, step)

    data = computePhaseData(t.size, t, step)

    for i in range(0, t.size):
        for j in range(0, t.size):

            plt.plot(data[i, j, 0], data[i, j, 1], 'k.', markersize=0.5) #trajectoire
            #plt.plot(initial_values[0], initial_values[1], 'r.', markersize=2) #conditions initiales
            plt.plot(data[i, j, 0, -1], data[i, j, 1, -1], 'b.', markersize=4) #points finals
        
    plt.xlabel("déplacement (cm)")
    plt.ylabel("vitesse (cm/s)")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    main()
