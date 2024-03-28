import numpy as np
import time

k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s

# here no need for nH nor ne as they cancel out
def equilibrium1(T,Z,Tc,psi):
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k

def equilibrium2(T,Z,Tc,psi, nH, A, xi):
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)

def Brent_method(f, a, c, rtol=1e-5, atol=1e-8, maxiter=1000):
    """
    Brent's Method for minimalization
    f is the function to be minimized
    a and c are the endpoints of the initial bracket to search in
    """
    start = time.time()

    gr = 1/(1+((1+np.sqrt(5))*0.5)) # golden ratio

    epsilon_m = 2**-52 # machine epsilon assuming float64

    w, v, b = (a+c)/2, (a+c)/2, (a+c)/2

    for itt in range(maxiter):
        # Step 6
        if np.abs(a-c) < rtol*2*np.abs(b) or np.abs(b) < atol:
            print("Steps take: ",itt)
            print("Time taken: ", time.time()-start, " s")
            return b
        
        # Step 1
        if len(np.unique([w,v,b])) == 3:
            d = parabola_min(f, w, v, b)
        else:
            #print(np.unique([w,v,b]))
            # Golden Ratio Step
            if np.abs(c-b) > np.abs(b-a):
                d = b + (c-b)*gr
            else:
                d = b + (a-b)*gr
        
        # Step 2
        if a<d<c and np.abs(d-b) < np.abs(v-w):
            pass
        else:
            # Step 3
            # Golden Ratio Step
            # if np.abs(w-b) > np.abs(b-v):
            #     d = b + (w-b)*gr
            # else:
            #     d = b + (v-b)*gr
            if np.abs(c-b) > np.abs(b-a):
                d = b + (c-b)*gr
            else:
                d = b + (a-b)*gr
        
        # Step 4
        if np.abs(d-b) <= epsilon_m:
            d = b + np.sign(d-b)*epsilon_m

        # Step 5
        if d > b:
            if f(d) < f(b):
                a, b = b, d
            else:
                c = d
        else:
            if f(d) < f(b):
                c, b = b, d
            else:
                a = d

        b, w, v = d, b, w

    print("Steps take: ",itt)
    print("Time taken: ", time.time()-start, " s")
    return b

def parabola_min(f, x0, x1, x2):
    """
    This function fits a parabola through the three points (x0, f(x0)), (x1, f(x1)), (x2, f(x2)) and returns the x-value of the minimum of this parabola
    """
    y0 = f(x0)
    y1 = f(x1)
    y2 = f(x2)

    norm = (x0-x1)*(x0-x2)*(x1-x2)
    a = (x2 * (y1-y0) + x1 * (y0-y2) + x0 * (y2-y1)) / norm
    b = (x2**2 * (y0-y1) + x1**2 * (y2-y0) + x0**2 * (y1-y2)) / norm
    #c = (x1*x2*(x1-x2)*y0 + x2*x0*(x2-x0)*y1 + x0*x1*(x0-x1)*y2) / norm
    return -b/(2*a)


# 2a
# Minimization of photoionization-ratidative recombination to find equilibrium temperature
x = np.logspace(0, 7, 200)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, np.abs(equilibrium1(x, 0.015, 1e4, 0.929)), label='Z=1, Tc=1, psi=1')
plt.xscale('log')
plt.savefig('plots/2a.png')

T_equilibrium = Brent_method(lambda x: np.abs(equilibrium1(x, 0.015, 1e4, 0.929)), 1, 1e7)
print(f"Equilibrium temperature: {T_equilibrium:.2e} K")
with open('output/2a.txt', 'w') as f:
    f.write(f"Equilibrium temperature: {T_equilibrium} K")

# 2b
with open('output/2b.txt', 'w') as f:
    for n_e in [1e-4, 1, 1e4]:
        T_equilibrium2 = Brent_method(lambda x: np.abs(equilibrium2(x, 0.015, 1e4, 0.929, n_e, 5e-10, 1e-15)), 1, 1e15, rtol=1e-10, atol=1e-50)
        print(f"Equilibrium temperature for n_e = {n_e:.2e} cm^-3: {T_equilibrium2} K")
        f.write(f"Equilibrium temperature for n_e = {n_e:.2e} cm^-3: {T_equilibrium2} K\n")

x = np.logspace(1,15,100)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize=(10,5))
for n_e in [1e-4, 1, 1e4]:
    ax[0].plot(x, np.abs(equilibrium2(x, 0.015, 1e4, 0.929, n_e, 5e-10, 1e-15)), label='n_e = '+str(n_e))
x = np.linspace(1e14,2e15,100)
for n_e in [1e-4, 1, 1e4]:
    ax[1].plot(x, np.abs(equilibrium2(x, 0.015, 1e4, 0.929, n_e, 5e-10, 1e-15)), label='n_e = '+str(n_e))
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].legend()
ax[1].legend()
fig.savefig('plots/2b.png')