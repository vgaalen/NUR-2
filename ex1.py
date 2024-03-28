import numpy as np
import matplotlib.pyplot as plt

def Romberg_integration(f,a,b,*args,m=5,h=1):
    """
    Implementation of Romberg Integration.
    Numerical integration of function f(x), from x=a until x=b
    """
    if a>b:
        a,b=b,a
    
    h=h*(b-a)

    r = np.zeros(m)
    r[0] = 0.5*h*(f(a,*args)+f(b,*args))

    N_p = 1
    for i in range(1,m):
        delta = h
        h = 0.5*h
        x = a + h

        for j in range(N_p):
            r[i] += f(x,*args)
            x += delta
        
        r[i] = 0.5*(r[i-1]+delta*r[i])
        N_p *= 2
    
    N_p = 1
    for i in range(1,m):
        N_p *= 4
        for j in range(m-i):
            r[j] = (N_p*r[j+1]-r[j])/(N_p-1)
    
    return r[0]

def n(x,A,Nsat,a,b,c):
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def spherical_integral_n(x,A,Nsat,a,b,c):
    return 4*np.pi * A * Nsat * x**(a-1) * b**(3-a) * np.exp(-(x/b)**c)


A=1. # to be computed
Nsat=100
a=2.4
b=0.25
c=1.6

print(f"Integral of n(x): {Romberg_integration(spherical_integral_n,0,5,A,Nsat,a,b,c)}")
A = A*Nsat / (Romberg_integration(spherical_integral_n,0,5,A,Nsat,a,b,c))
print(f"Normalization Factor A: {A}")
with open('1a.txt', 'w') as f:
    f.write(f'{A}\n')
print(f"Integral of n(x): {Romberg_integration(spherical_integral_n,0,5,A,Nsat,a,b,c)}")



# 1b
class RNG():
    def __init__(self, seed):
        self.seed = seed
        self.x = np.uint64(seed)
        self.a1 = np.uint64(21)
        self.a2 = np.uint64(35)
        self.a3 = np.uint64(4)

        self.a = np.uint64(1664525)
        self.c = np.uint64(1013904223)
        #self.a = np.uint64(5478)
        #self.c = np.uint64(13912)
    
    def __call__(self, shape=None):
        """
        Generate random numbers using XOR_shift and LCG between 0 and 1
        """
        if shape is None:
            return self._LCG(self._XOR_shift())/18446744073709551615
        else:
            return np.array([self._LCG(self._XOR_shift())/18446744073709551615 for _ in range(np.prod(shape))]).reshape(shape)
    
    def get_state(self):
        return {"seed": self.seed, "a1": self.a1, "a2": self.a2, "a3": self.a3, "a": self.a, "c": self.c}

    def _XOR_shift(self):
        """
        Implementation of XOR_shift Random Number Generator
        """
        x = self.x
        x = x ^ ( x >> self.a1 )
        x = x ^ ( x << self.a2 )
        x = x ^ ( x >> self.a3 )
        self.x = x
        return x
    
    def _LCG(self, seed):
        """
        Implementation of Linear Congruential Generator
        """
        x = (self.a*seed + self.c) % 2**64
        return x

def transformation_function(x):#, A, Nsat, a, b, c):
    """
    Function that transforms the random numbers to the distribution of phi on a uniform sphere
    P(y) = int_0^y n(x) dx

    Parameters:
    x: float or numpy array
        Random numbers between 0 and 1
    """
    return np.arccos(1-2*x)

def rejection_sampling(p, rng, N):
    """
    Rejection sampling algorithm

    Parameters:
    p: function
        the probability density function
    rng: RNG
        random number generator, that gives random numbers between 0 and 1
    N: int
        number of samples to generate
    """
    samples = []
    while len(samples) < N:
        x = 5*rng()
        y = rng()
        if y <= p(x):
            samples.append(x)
    return np.array(samples)


#Plot of histogram in log-log space with line (question 1b)
xmin, xmax = 10**-4, 5
N_generate = 10000

rng = RNG(34906)

def prob_density(x):
    return spherical_integral_n(x, A, Nsat, a, b, c)/Nsat

# Generate a set of satellite positions
r = rejection_sampling(prob_density, rng, N_generate)
phi = transformation_function(rng(shape=[N_generate]))
theta = rng(shape=[N_generate])*2*np.pi

plt.figure()
edges = np.linspace(0,np.pi,21)
hist = np.histogram(np.sort(phi), bins=edges)[0]
plt.stairs(hist/np.max(hist), edges=edges, fill=True, label='Distribution in phi')
plt.plot(edges, np.sin(edges), 'r-', label='Analytical function')
plt.title('Review of Transformation Method on phi-coordinates')
plt.savefig('plots/phi.png', dpi=1000)

plt.figure()
edges = np.linspace(0,2*np.pi,21)
hist = np.histogram(np.sort(theta), bins=edges)[0]
plt.stairs(hist, edges=edges, fill=True, label='Distribution in theta')
plt.plot(edges[:-1]+np.diff(edges)/2, [N_generate/20]*20, 'r-', label='Analytical function')
plt.title('Review of Transformation Method on theta-coordinates')
plt.savefig('plots/theta.png', dpi=1000)

#21 edges of 20 bins in log-space
edges = 10**np.linspace(np.log10(xmin), np.log10(xmax), 21)
hist = np.histogram(np.sort(r), bins=edges)[0] #replace!

relative_radius = edges[:-1] + np.diff(edges)/2
analytical_function = [Romberg_integration(spherical_integral_n,low_bound,high_bound,A,Nsat,a,b,c) for low_bound, high_bound in zip(edges[:-1], edges[1:])]

fig1b, ax = plt.subplots()
ax.stairs(hist *Nsat/N_generate, edges=edges, fill=True, label='Satellite galaxies')
plt.plot(relative_radius, analytical_function, 'r-', label='Analytical solution') #correct this according to the exercise!
ax.set(xlim=(xmin, xmax), ylim=(10**(-3), 100), yscale='log', xscale='log',
       xlabel='Relative radius', ylabel='Number of galaxies')
ax.legend()
plt.savefig('plots/my_solution_1b.png', dpi=600)



# 1c
# Select 100 random galaxies from the previous sample
def selection_sort(set, N):
    """
    Variation of the Selection Sort algorithm that only finds the smallest N elements
    The input array is sorted using the values in the first column

    Parameters
    ----------
    set :ndarray
        The array to sort
    N: int
        Number of elements to find

    Returns
    -------
    ndarray
        The input array with the N smallest elements in the first N positions
    """
    for i in range(N):
        min = i
        for j in range(i+1, len(set[:,0])):
            if set[j,0] < set[min,0]:
                min = j
        set[i,0], set[min,0] = set[min,0], set[i,0]
        set[i,1], set[min,1] = set[min,1], set[i,1]
    return set


#Cumulative plot of the chosen galaxies (1c)
chosen = selection_sort(np.array([rng(shape=[N_generate]), r]).T, 100)[:100,1]

fig1c, ax = plt.subplots()
ax.plot(np.sort(chosen), np.arange(100))
ax.set(xscale='log', xlabel='Relative radius', 
       ylabel='Cumulative number of galaxies',
       xlim=(xmin, xmax), ylim=(0, 100))
plt.savefig('plots/my_solution_1c.png', dpi=600)



# 1d
def Ridder(f, x, m, target_err, h_init, d=0.5):
    """
    Ridder's Method for derivatives

    Parameters
    ----------
    f : function
        The function to differentiate
    x : float
        The x value to differentiate at
    m : int
        The order of the derivative
    target_err : float
        The desired accuracy
    h_init : float
        The initial step size
    d : float
        The step size reduction factor
    """

    # Check if m is a positive integer
    if m < 0 or not isinstance(m, int):
        raise ValueError('m must be a positive integer')

    # Check if h_init is positive
    if h_init <= 0:
        raise ValueError('h_init must be positive')
    
    # Check if d is between 0 and 1
    if d <= 0 or d >= 1:
        raise ValueError('d must be between 0 and 1')
    
    # Check if target_err is positive
    if target_err <= 0:
        raise ValueError('target_err must be positive')
    
    D = np.zeros((m+1,m+1))
    h = h_init
    for i in range(m+1):
        D[i,0] = (f(x+h)-f(x-h))/(2*h)
        h = h * d

    for j in range(1, m+1):
        for i in range(0, m+1-j):
            ##D[j] = (4**i*D[j]-D[j-1])/(4**i-1)
            D[i,j] = (d**(2*j) * D[i+1,j-1]-D[i,j-1])/(d**(2*j)-1)

            if np.abs(D[i,j]-D[i,j-1]) < target_err:
                return D[i,j]
            if j>1 and np.abs(D[i,j]-D[i,j-1])>np.abs(D[i,j-1]-D[i,j-2]):
                return D[i,j-1]

    return Ridder(f, x, m+1, target_err, h_init, d)

def dn_dx(x, A, Nsat, a, b, c):
    return A*Nsat*(x/b)**(a-4)*np.exp(-(x/b)**c)/b * ( -c*(x/b)**c + (a-3) )


numerical_derivative = Ridder( lambda x: n(x, A, Nsat, a, b, c), 1, 0, 1e-15, 0.5, d=0.5)
analytical_derivative = dn_dx(1, A, Nsat, a, b, c)

print("Numerical derivative: ", numerical_derivative)
print("Analytical derivative: ", analytical_derivative)
print("Difference: ", np.abs(numerical_derivative-analytical_derivative))

with open('1d.txt', 'w') as f:
    f.write(f'Numerical Derivative (Ridder\'s Method): {numerical_derivative}\n')
    f.write(f'Analytical Derivative: {analytical_derivative}\n')
    f.write(f'Difference between these: {np.abs(numerical_derivative-analytical_derivative)}\n')