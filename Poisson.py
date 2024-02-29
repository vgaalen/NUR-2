

def poisson_dist(lambd, k):
    val = np.float32((2.71828**(-lambd)))
    while k > 1:
        val /= k
        val *= lambd
        k -= 1
    return val

def poisson_dist_log(lambd, k):
    return np.exp(k*np.log(lambd) - lambd - np.sum(np.log(np.arange(1,k+1))))

if __name__=='__main__':
    import numpy as np

    ## 1a ##
    k_values = np.array( [0, 10, 21, 40, 200], dtype=np.int32)
    lambda_values = np.array( [1, 5, 3, 2.6, 101], dtype=np.float32)
    output = np.zeros(len(k_values), dtype=np.float32)

    for i in range(len(k_values)):
        output[i] = poisson_dist_log(lambda_values[i], k_values[i])
    
    with open("1a.txt", 'w') as f:
        f.write("(\lambda, k), Poisson Probability Distribution\n")
        for i in range(len(k_values)):
            f.write(f"({lambda_values[i]}, {k_values[i]}) {poisson_dist(lambda_values[i], k_values[i])} ")
            f.write("\n")

    from timeit import timeit
    t1 = timeit("""for i in range(len(k_values)):
        output[i] = poisson_dist(lambda_values[i], k_values[i])""", globals=globals(), number=1000)
    t2 = timeit("""for i in range(len(k_values)):
        output[i] = poisson_dist_log(lambda_values[i], k_values[i])""", globals=globals(), number=1000)
    
    with open("poisson_timing.txt", 'w') as f:
        f.write(f"Itterative Approach: {t1}s \nlog-space Approach: {t2}s")