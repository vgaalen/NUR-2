import numpy as np

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


def test_Romberg():
    def function(x):
        return x**2
    def function2(x):
        return np.exp(x)

    print(f"Romber_int: ", Romberg_integration(function,1,2))
    print(f"Romber_int: ", Romberg_integration(function2,0,5))
    assert Romberg_integration(function,1,2) == 2.3333333333333335
    assert Romberg_integration(function2,0,5) - 147.41315910257657 < 1e-4


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

#test_Romberg()

A=1. # to be computed
Nsat=100
a=2.4
b=0.25
c=1.6

def n(x,A,Nsat,a,b,c):
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def spherical_integral_n(x,A,Nsat,a,b,c):
    return 4*np.pi * A * Nsat * x**(a-1) * b**(3-a) * np.exp(-(x/b)**c)

print(f"Integral of n(x): {Romberg_integration(spherical_integral_n,0,5,A,Nsat,a,b,c)}")
A = A*Nsat / (Romberg_integration(spherical_integral_n,0,5,A,Nsat,a,b,c))
print(f"Normalization Factor A: {A}")
with open('1a.txt', 'w') as f:
    f.write(f'{A}\n')
print(f"Integral of n(x): {Romberg_integration(spherical_integral_n,0,5,A,Nsat,a,b,c)}")

# Generate a set of satellite positions
def transformation_function(x, A, Nsat, a, b, c):
    """
    Function that transforms the random numbers to the distribution of n(x)
    P(y) = int_0^y n(x) dx

    Parameters:
    x: float or numpy array
        Random numbers between 0 and 1
    """
    if isinstance(x, np.ndarray):
        return np.array([Romberg_integration(spherical_integral_n, 0, x_i, A, Nsat, a, b, c)/Nsat for x_i in x])
    elif isinstance(x, float):
        return Romberg_integration(spherical_integral_n, 0, x, A, Nsat, a, b, c)/Nsat
    else:  
        raise ValueError("x must be either a float or a numpy array")

#Plot of histogram in log-log space with line (question 1b)
xmin, xmax = 10**-4, 5
N_generate = 10000

rng = RNG(34906)

plt.figure()
plt.plot([1]*50, rng(shape=[50]), '.')
plt.savefig('test.png', dpi=1000)


x = 5*rng(shape=[N_generate])

r = transformation_function(x, A, Nsat, a, b, c)
# print("Transformation function min: ", np.min(r))
# print("Transformation function max: ", np.max(r))
# print(r[r>1].size)

phi = rng(shape=[N_generate])*2*np.pi
theta = rng(shape=[N_generate])*np.pi

#21 edges of 20 bins in log-space
edges = 10**np.linspace(np.log10(xmin), np.log10(xmax), 21)
hist = np.histogram(np.sort(r), bins=edges)[0] #replace!

relative_radius = edges[:-1] + np.diff(edges)/2
#analytical_function = spherical_integral_n(relative_radius, A, Nsat, a, b, c) #replace

analytical_function = [Romberg_integration(spherical_integral_n,low_bound,high_bound,A,Nsat,a,b,c) for low_bound, high_bound in zip(edges[:-1], edges[1:])]
#/(high_bound-low_bound)
print(relative_radius)
print(analytical_function)
print("Max Analytical function: ", np.max(analytical_function))


fig1b, ax = plt.subplots()
ax.stairs(hist*np.diff(edges) * Nsat/N_generate, edges=edges, fill=True, label='Satellite galaxies')
plt.plot(relative_radius, analytical_function, 'r-', label='Analytical solution') #correct this according to the exercise!
ax.set(xlim=(xmin, xmax), ylim=(10**(-3), 100), yscale='log', xscale='log',
       xlabel='Relative radius', ylabel='Number of galaxies')
ax.legend()
plt.savefig('my_solution_1b.png', dpi=600)

# Select 100 random galaxies from the previous sample
def select(set, rng):
    ind = int(rng()*len(set))
    return set[ind], ind

def take_subset(set, num):
    rng = RNG(1762)
    subset = np.zeros(num)
    for i in range(num):
        subset[i], ind = select(set, rng)
        set = np.delete(set, ind)
    return subset


#Cumulative plot of the chosen galaxies (1c)
chosen = xmin + np.sort(take_subset(r.copy(), 100))*(xmax-xmin) #replace!
fig1c, ax = plt.subplots()
ax.plot(chosen, np.arange(100))
ax.set(xscale='log', xlabel='Relative radius', 
       ylabel='Cumulative number of galaxies',
       xlim=(xmin, xmax), ylim=(0, 100))
plt.savefig('my_solution_1c.png', dpi=600)


