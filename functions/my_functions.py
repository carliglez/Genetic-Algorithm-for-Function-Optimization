import numpy as np

# Define the Alluffi-Pentiny function
def alluffi_pentiny(x):
    return 1/4 * x[0]**4 - 1/2 * x[0]**2 + 1/10 * x[0] + 1/2 * x[1]**2

# Define the Bohachevsky 1 function
def bohachevsky_1(x):
    return x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7

# Define the Bohachevsky 2 function
def bohachevsky_2(x):
    return x[0]**2 + 2 * x[1]**2 - 0.3 * np.cos(3 * np.pi * x[0]) * np.cos(4 * np.pi * x[1]) + 0.3

# Define the Becker and Lago function
def becker_lago(x):
    return (np.abs(x[0]) - 5)**2 + (np.abs(x[1]) - 5)**2

# Define the Branin function
def branin(x):
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    u = 10 * (1 - t)
    
    return (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + u

# Define the Camel function
def camel(x):
    return 4 * x[0]**2 - 2.1 * x[0]**4 + (x[0]**6)/3 + x[0]*x[1] - 4 * x[1]**2 + 4 * x[1]**4

# Define the Cb3 function (Three-Hump Camel function)
def cb3(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2

# Define the Cosine Mixture (CM) function
def cosine_mixture(x):
    return np.sum(x**2) - 0.1 * np.sum(np.cos(5 * np.pi * x))

# Define the DeJong function
def dejong(x):
    return np.sum(x**2)

# Define the updated Easom function
def easom(x):
    try:
        exp_term = np.exp(-((x[1] - np.pi)**2 - (x[0] - np.pi)**2) / 1e6)
        result = -np.cos(x[0]) * np.cos(x[1]) * exp_term
        return result
    except OverflowError:
        return -np.inf
    # We divide the large exponents by a constant (1e6) to reduce their magnitudes. This should help avoid overflow issues

# Define the Exponential function
def exponential(x):
    return -np.exp(-0.5 * np.sum(x**2))

# Define the GKLS function
def gkls(x, n, w):
    if len(x) != n:
        raise ValueError("Input vector size should be equal to n.")
    if not (2 <= n <= 100):
        raise ValueError("n should be a positive integer between 2 and 100.")
    
    # Parameters for Gkls function
    a = 1000.0
    r = 1.0
    s = 1.0
    m = 1.0
    
    result = 0.0
    for i in range(w):
        ai = a / (10 ** (i // n))
        p = (i % n) + 1
        result += ai * np.exp(-r * np.sum((x - (s * np.sin(m * np.pi * p / n))) ** 2))
    
    return -result / 1000.0  # Scaling factor

# Define the Goldstein and Price function
def goldstein_price(x):
    if len(x) != 2:
        raise ValueError("Goldstein and Price function is defined for 2-dimensional vectors.")
    
    x1, x2 = x
    term1 = 1 + ((x1 + x2 + 1)**2) * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + ((2*x1 - 3*x2)**2) * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    
    return term1 * term2

# Define the Griewank2 function
def griewank2(x):
    n = len(x)
    sum_term = np.sum(x**2) / 200
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return 1 + sum_term - prod_term

# Define the Hansen function
def hansen(x):
    term1 = np.sum([i * np.cos((i - 1) * x[0] + i) for i in range(1, 6)])
    term2 = np.sum([j * np.cos((j + 1) * x[1] + j) for j in range(1, 6)])
    return term1 * term2

# Define the Hartman 3 function
def hartman_3(x):
    # Define coefficients
    a = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    c = np.array([1, 1.2, 3, 3.2])
    p = np.array([[0.3689, 0.117, 0.2673],
                  [0.4699, 0.4387, 0.747],
                  [0.1091, 0.8732, 0.5547],
                  [0.03815, 0.5743, 0.8828]])

    # Compute the function value
    result = -np.sum(c * np.exp(-np.sum(a * (x - p)**2, axis=1)))

    return result

# Define the Hartman 6 function
def hartman_6(x):
    # Define coefficients
    a = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    c = np.array([1, 1.2, 3, 3.2])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                  [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                  [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

    # Compute the function value
    result = -np.sum(c * np.exp(-np.sum(a * (x - p)**2, axis=1)))

    return result

# Define the Rastrigin function
def rastrigin(x):
    result = x[0]**2 + x[1]**2 - np.cos(18 * x[0]) - np.cos(18 * x[1])
    return result

# Define the Rosenbrock function
def rosenbrock(x):
    n = len(x)
    result = sum(100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(n - 1))
    return result

# Define the Shekel 5 parameters
a_shekel = np.array([[4, 4, 4, 4],
                     [1, 1, 1, 1],
                     [8, 8, 8, 8],
                     [6, 6, 6, 6],
                     [3, 7, 3, 7]])

c_shekel = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
# Define the Shekel 5 function
def shekel_5(x):
    shekel_sum = 0
    for i in range(5):
        shekel_sum -= 1 / (np.sum((x - a_shekel[i])**2) + c_shekel[i])
    return shekel_sum

# Define Shekel 7 parameters
a_shekel7 = np.array([[4, 4, 4, 4],
                      [1, 1, 1, 1],
                      [8, 8, 8, 8],
                      [6, 6, 6, 6],
                      [3, 7, 3, 7],
                      [2, 9, 2, 9],
                      [5, 3, 5, 3]])

c_shekel7 = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])
# Define the Shekel 7 function
def shekel_7(x):
    result = -np.sum(1 / (np.sum((x - a_shekel7)**2, axis=1) + c_shekel7))
    return result

# Define Shekel 10 parameters
a_shekel10 = np.array([[4, 4, 4, 4],
                       [1, 1, 1, 1],
                       [8, 8, 8, 8],
                       [6, 6, 6, 6],
                       [3, 7, 3, 7],
                       [2, 9, 2, 9],
                       [5, 5, 3, 3],
                       [8, 1, 8, 1],
                       [6, 2, 6, 2],
                       [7, 3.6, 7, 3.6]])

c_shekel10 = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.6])
# Define the Shekel 10 function
def shekel10(x):
    result = -np.sum(1 / (np.sum((x - a_shekel10)**2, axis=1) + c_shekel10))
    return result

# Define the Shubert function 
def shubert(x):
    result = -np.sum(np.array([np.sum((j + 1) * (np.sin((j + 1) * x_i) + 1)) for j, x_i in enumerate(x)]))
    return result

# Define the Sinusoidal function
def sinusoidal(x, z):
    term1 = 2.5 * np.prod(np.sin(x - z))
    term2 = np.prod(np.sin(5 * (x - z)))
    result = -(term1 + term2)
    return result

# Define the Test2N function 
def test2n(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)

# Define the Test30N function
def test30n(x):
    n = len(x)
    return (1/10) * np.sin(3 * np.pi * x[0])**2 * np.sum((x[1:n-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[2:n])**2)) + (x[n-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[n-1])**2)

# Define the Potential function
def potential(x):
    # Lennard-Jones potential parameters
    sigma = 1.0
    epsilon = 1.0
    
    # Reshape the input vector to get the atomic coordinates
    coordinates = x.reshape((-1, 3))
    
    # Calculate the energy using the Lennard-Jones potential
    energy = 0.0
    n = len(coordinates)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = np.linalg.norm(coordinates[i] - coordinates[j])
            
            # Check for very small distances to avoid division by zero
            if rij < 1e-12:
                continue
            
            energy += 4 * epsilon * ((sigma / rij) ** 12 - (sigma / rij) ** 6)
    
    return energy