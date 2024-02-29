import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit

def LU_decomp(mat):
    """
    Perform a LU decomposition of a matrix.
    Both the L and U matrices are stored in the same matrix, with all values below the diagonal representing the U matrix
    and the diagonal and above representing the L matrix (with the U matrix's diagonal consisting of 1's and the other missing values being 0's).
    """
    mat[1:,0] = mat[1:,0] / mat[0,0]
    for j in range(1,mat.shape[1]):
        for i in range(1,mat.shape[0]):
            if i <= j:
                mat[i,j] = mat[i,j] - np.sum(np.multiply(mat[i,:i],mat[:i,j]))
            else:
                mat[i,j] = (mat[i,j] - np.sum(np.multiply(mat[i,:j],mat[:j,j]))) / mat[j,j]
    return mat

def LU_solve(LU, y):
    """
    Solve a system of linear equations using the LU decomposition of a matrix.
    """

    #y[0] = y[0]
    for i in range(1,len(y)):
        y[i] = (y[i] - np.sum(np.multiply(LU[i,:i],y[:i])))
    
    y[-1] = y[-1]/LU[-1,-1]
    for i in range(len(y)-2, -1, -1):
        y[i] = 1/LU[i,i]*(y[i] - np.sum(np.multiply(LU[i,i+1:],y[i+1:])))
    
    return y

# def LU_decomp_test():

#     #mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     #mat = np.array([[4,5,6],[7,8,9],[1,2,3]],dtype=np.float32)Q
#     mat = np.array([[1,2,1],[7,9,4],[3,8,5]],dtype=np.float32)
#     LU = LU_decomp(mat)
#     print("LU: ",LU)
#     L = np.tril(LU, -1) + np.eye(3)
#     U = np.triu(LU)
#     print("L: ",L)
#     print("U: ",U)
#     print("L@U: ",np.matmul(L,U))

#     #mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
#     #mat = np.array([[4,5,6],[7,8,9],[1,2,3]], dtype=np.float32)
#     mat = np.array([[1,2,1],[7,9,4],[3,8,5]],dtype=np.float64)
#     print(LU_solve(mat.copy(), np.array([1,1,1],dtype=np.float32)))
#     print(mat@LU_solve(mat.copy(), np.array([25,74,91],dtype=np.float64)))


def TwoA(data):
    ##############
    ##### 2a #####
    ##############
    
    data.astype(np.float128)
    matrix = np.repeat([data[:,0]], data.shape[0], axis=0)
    power = np.arange(data.shape[0], dtype=np.float128)
    powers = np.repeat([power], data.shape[0], axis=0)
    matrix = np.power(matrix, powers.T).astype(np.float128)

    LU = LU_decomp(matrix.copy().T)
    c = LU_solve(LU, data[:,1].astype(np.float128).copy())

    print(c)
    np.savetxt('vandermonde_coefficients.txt', c, delimiter=' ')

    x = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 1000)
    powers = np.repeat([power], 1000, axis=0)
    xx = np.repeat([x], data.shape[0], axis=0)
    cc = np.repeat([c], 1000, axis=0)
    y = np.sum( np.multiply(cc.T, np.power(xx, powers.T)), axis=0 )

    fig, ax = plt.subplots(1, 2, figsize=(10,5), dpi=192)
    ax[0].plot(data[:,0], data[:,1], 'ro', label="datapoints")
    ax[1].plot(data[:,0], data[:,1], 'ro', label="datapoints")
    ax[0].plot(x, y, 'b', label='LU decomposition')
    ax[1].plot(x, y, 'b', label='LU decomposition')
    for i in range(data.shape[0]):
        #ax[0].text(data[i,0], data[i,1]+250, str(f"{data[i,1]-y[np.argmin(np.abs(x-data[i,0]))]:.2f}"), fontdict={'size': 5})
        #ax[1].text(data[i,0], data[i,1]+10, str(f"{data[i,1]-y[np.argmin(np.abs(x-data[i,0]))]:.2f}"), fontdict={'size': 5})
        ax[0].text(data[i,0]-4, data[i,1]+(-1)**i*250, \
                   np.format_float_scientific(np.abs(data[i,1]-np.sum( np.multiply(c.T, np.power(data[i,0], power)), axis=0 )), precision=2), fontdict={'size': 5})
        ax[1].text(data[i,0]-4, data[i,1]+10, \
                   np.format_float_scientific(np.abs(data[i,1]-np.sum( np.multiply(c.T, np.power(data[i,0], power)), axis=0 )), precision=2), fontdict={'size': 5})

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].set_ylabel('y')
    fig.suptitle('Solution to Vandermonde Matrix')
    ax[1].set_ylim([np.min(data[:,1])-50, np.max(data[:,1])+50])
    ax[0].legend()
    ax[1].legend()
    #plt.yscale('log')
    fig.tight_layout()
    fig.savefig('vandermonde.png')
    return fig, ax



def bisection(x, x0, num=2):
    """
    Bisection algorithm for finding the index of the reference data point that x is between

    Parameters
    ----------
    x : float
        The x value to interpolate at
    x0 : float
        The x values of the reference data points
    """
    if x < x0[0]:
        return [0]
    elif x > x0[-1]:
        return [len(x0)-1]
    indices = range(len(x0))
    while len(indices) > 2:
        #print(len(indices))
        if x > x0[indices[len(indices)//2]]:
            indices = indices[len(indices)//2:]
        else:
            indices = indices[:len(indices)//2+1]
    if num==2:
        return indices
    elif num==1:
        closest = np.argmin(np.abs(x-x0[indices]))
        return indices[closest]

def neville(x, x0, y0, ord=None, fill_value=None):
    """
    Neville's algorithm for polynomial interpolation

    Parameters
    ----------
    x : float
        The x value to interpolate at
    x0 : float
        The x values of the reference data points
    y0 : float
        The y values of the reference data points
    ord : int, optional
        The order of the polynomial to interpolate with. If None, the order is set to len(x0)-1
    fill_value : float, optional
        The value to return if x is outside the range of x0. If None, an error is raised.
    """

    if ord==None:
        ord = len(x0)-1
    # if x < np.min(x0) or x > np.max(x0):
    #     if fill_value is not None:
    #         return fill_value
    #     else:
    #         raise ValueError('x is out of range')
    
    p = np.zeros((ord,ord))
    x_i = np.zeros((ord))

    for i in range(ord):
        closest = bisection(x, x0, num=1)
        x_i[i] = x0[closest]
        x0 = np.delete(x0,closest,0)
        p[i,i] = y0[closest]
        y0 = np.delete(y0,closest,0)
    
    for k in range(1,ord):
        for i in range(ord-k):
            j = i+k
            p[i,j] = ( (x-x_i[i])*p[i+1,j] - (x-x_i[j])*p[i,j-1] ) / (x_i[j]-x_i[i])
    return p[0,ord-1]



def TwoB(data, fig=None, ax=None):
    ##############
    ##### 2b #####
    ##############
    x = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 1000)
    y_neville = np.zeros((len(x)))
    for i in range(len(x)):
        y_neville[i] = neville(x[i], data[:,0], data[:,1])

    if fig is not None and ax is not None:
        #fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi=192)
        #ax[0].plot(data[:,0], data[:,1], 'ro', label='Data points')
        #ax[1].plot(data[:,0], data[:,1], 'ro', label='Data points')
        
        ax[0].plot(x, y_neville, 'g', label="Neville's Algorithm")
        ax[1].plot(x, y_neville, 'g', label="Neville's Algorithm")

        for i in range(data.shape[0]):
            ax[0].text(data[i,0]-4, data[i,1]+(-1)**i*250, \
                    np.format_float_scientific(np.abs(data[i,1]-neville(data[i,0], data[:,0], data[:,1])), precision=2), fontdict={'size': 5})
            ax[1].text(data[i,0]-4, data[i,1]+10, \
                    np.format_float_scientific(np.abs(data[i,1]-neville(data[i,0], data[:,0], data[:,1])), precision=2), fontdict={'size': 5})
        
        ax[0].set_xlabel('x')
        ax[1].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[1].set_ylabel('y')
        fig.suptitle("Neville's Algorithm")
        ax[1].set_ylim([np.min(data[:,1])-50, np.max(data[:,1])+50])
        ax[0].legend()
        ax[1].legend()
        #plt.yscale('log')
        fig.tight_layout()
        plt.savefig('neville_compare.png')
        plt.close()

def TwoC(data):
    ##############
    ##### 2c #####
    ##############

    # Iterative Improvements on LU decomposition
    # first itteration gives c' in V@c=y with c'=c+dc, where dc is the error
    # second itteration we solve for V@dc=y-V@c', subtracting dc from c' then gives c''
    # and so on

    matrix = np.repeat([data[:,0]], data.shape[0], axis=0)
    power = np.arange(data.shape[0], dtype=np.float128)
    powers = np.repeat([power], data.shape[0], axis=0)
    matrix = np.power(matrix, powers.T).astype(np.float128)

    LU = LU_decomp(matrix.copy().T)
    c = LU_solve(LU, data[:,1].astype(np.float128).copy())

    #LU1 = c - LU_solve(matrix.copy().T, data[:,1].copy()-np.sum( np.multiply(c.T, np.power(data[:,0], power)), axis=0 ))
    LU1 = c - LU_solve(LU, matrix.T@c-data[:,1].copy())

    LU10 = LU1
    for i in range(9):
        LU10 = LU10 - LU_solve(LU, matrix.T@LU10-data[:,1].copy())

    fig, ax = plt.subplots(1, 3, figsize=(15,5), dpi=192)
    ax[0].plot(data[:,0], data[:,1], 'ro', label="datapoints")
    ax[1].plot(data[:,0], data[:,1], 'ro', label="datapoints")
    ax[2].plot(data[:,0], data[:,1], 'ro', label="datapoints")
    
    x = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 1000)
    x_matrix = np.repeat([x], data.shape[0], axis=0)

    power = np.arange(data.shape[0], dtype=np.float128)
    powers = np.repeat([power], x.shape[0], axis=0)
    x_matrix = np.power(x_matrix, powers.T).astype(np.float128)

    y = x_matrix.T@c
    ax[0].plot(x,y, 'b', label='LU0')

    y = x_matrix.T@LU1
    ax[1].plot(x, y, 'b', label='LU1')

    y = x_matrix.T@LU10
    ax[2].plot(x, y, 'b', label='LU10')

    val_LU0 = matrix.T@c
    val_LU1 = matrix.T@LU1
    val_LU10 = matrix.T@LU10
    text = []
    for i in range(data.shape[0]):
        #ax[0].text(data[i,0], data[i,1]+250, str(f"{data[i,1]-y[np.argmin(np.abs(x-data[i,0]))]:.2f}"), fontdict={'size': 5})
        #ax[1].text(data[i,0], data[i,1]+10, str(f"{data[i,1]-y[np.argmin(np.abs(x-data[i,0]))]:.2f}"), fontdict={'size': 5})
        text.append(ax[0].text(data[i,0]-4, data[i,1]+(-1)**i*250, \
                   np.format_float_scientific(np.abs(data[i,1]-val_LU0[i]), precision=2), fontdict={'size': 5}))
        text.append(ax[1].text(data[i,0]-4, data[i,1]+(-1)**i*250, \
                   np.format_float_scientific(np.abs(data[i,1]-val_LU1[i]), precision=2), fontdict={'size': 5}))
        text.append(ax[2].text(data[i,0]-4, data[i,1]+(-1)**i*250, \
                   np.format_float_scientific(np.abs(data[i,1]-val_LU10[i]), precision=2), fontdict={'size': 5}))

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[2].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].set_ylabel('y')
    ax[2].set_ylabel('y')
    fig.suptitle('Itterating on Vandermonde Matrix Solution')
    #ax[1].set_ylim([np.min(data[:,1])-50, np.max(data[:,1])+50])
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    #ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    fig.tight_layout()
    fig.savefig('vandermonde_itt.png')

    for t in text:
        t.remove()
    for i in range(data.shape[0]):
        text.append(ax[0].text(data[i,0]-4, data[i,1]+25, \
                   np.format_float_scientific(np.abs(data[i,1]-val_LU0[i]), precision=2), fontdict={'size': 5}))
        text.append(ax[1].text(data[i,0]-4, data[i,1]+25, \
                   np.format_float_scientific(np.abs(data[i,1]-val_LU1[i]), precision=2), fontdict={'size': 5}))
        text.append(ax[2].text(data[i,0]-4, data[i,1]+25, \
                   np.format_float_scientific(np.abs(data[i,1]-val_LU10[i]), precision=2), fontdict={'size': 5}))

    ax[0].set_ylim([np.min(data[:,1])-50, np.max(data[:,1])+50])
    ax[1].set_ylim([np.min(data[:,1])-50, np.max(data[:,1])+50])
    ax[2].set_ylim([np.min(data[:,1])-50, np.max(data[:,1])+50])
    fig.savefig('vandermonde_itt_zoom.png')

    plt.close()


if __name__=='__main__':
    data = np.loadtxt("Vandermonde.txt")
    fig, ax = TwoA(data)
    TwoB(data, fig=fig, ax=ax)
    TwoC(data)

    ##############
    ##### 2d #####
    ##############
    t_2a = timeit('TwoA(data)', globals=globals(), number=10)
    t_2b = timeit('TwoB(data)', globals=globals(), number=10)
    t_2c = timeit('TwoC(data)', globals=globals(), number=10)

    with open('vandermonde_timing.txt', 'w') as f:
        f.write(f"2a: {t_2a:.2e} s\n2b: {t_2b:.2e} s\n2c: {t_2c:.2e} s\n")
    

    
