#NLA:PROJECT 1
#DAFNI TZIAKOURI

#C1
#Let's install all the packages we are going to use.

import time
import timeit
import random
import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import ldl, solve_triangular, cholesky, lu_factor, lu_solve


#Firstly, we define the Newton Step algorithm to compute a for the step-size correction substeps:

def Newton_step(lamb0, dlamb, s0, ds):

    alpha = 1
    index_lamb0 = np.array(np.where(dlamb < 0))

    if index_lamb0.size > 0:

        alpha = min(alpha,np.min(-lamb0[index_lamb0]/dlamb[index_lamb0]))
    index_s0 = np.array(np.where(ds<0))

    if index_s0.size > 0:

        alpha = min(alpha, np.min(-s0[index_s0]/ds[index_s0]))

    return alpha

#C2
#We will create the Karush-Kuhn-Tucker (KKT) matrix, the lambdas and S matrices.

def Matrix_KKT(G, C, n, m, lamb, s):
    #arg:
    #G : Coefficients for the quadratic objective function.
    #C : Constraint matrix.
    #n : Number of variables.
    #m  Number of constraints.
    #lamb : Lagrange multipliers.
    #s : Slack variables.

    S = np.diag(s)
    Lambdas = np.diag(lamb)
    eq1 = np.concatenate((G, -C, np.zeros((n, m))), axis = 1)
    eq2 = np.concatenate((np.transpose(-C), np.zeros((m, m)), np.identity(m)), axis = 1)
    eq3 = np.concatenate((np.zeros((m, n)), S, Lambdas), axis = 1)
    Mat = np.concatenate((eq1, eq2, eq3))

    #returns: The KKT matrix.
    #The diagonal matrix of slack variables (S).
    #The diagonal matrix of Lagrange multipliers (Lambdas)

    return Mat, S, Lambdas


#Now, we define the F and the F(z) function:

def F(x, G, g):
    #this function returns the value of the objective function.
    return 0.5 * np.transpose(x).dot(G).dot(x) + np.transpose(g).dot(x)


def F_z(x,lamb,s, G, g, C, d):

    eq1 = G.dot(x) + g - C.dot(lamb)
    eq2 = s + d - np.transpose(C).dot(x)
    eq3 = s * lamb
    #Returns a vector of equations representing KKT conditions.
    return np.concatenate((eq1, eq2, eq3))


#C3
#The following algorithm solves F(z)=0 using the functions defined above and a modified Newton's method.

def Func_C3(n, maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "Yes"):
    np.random.seed(2) # Set the seed for random number generation
    random.seed(2)  # Set the seed for the random module

    # We define all the parameters we will need

    m = 2*n
    x = np.zeros((n))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x, lamb, s))
    G = np.identity(n)
    C = np.concatenate((G, - G), axis = 1)
    d = np.full((m), - 10)
    e = np.ones((m))
    g = np.random.normal(0, 1, (n))

    # We count the time if asked for when calling the function

    if Print_Time == "Yes":

        Start = time.time()

    # Create Matrix_KKT, S and Lambdas with the previously defined function

    Mat, S, Lambdas = Matrix_KKT(G, C, n, m, lamb,s)

    for i in range(maxIter):

        b = -F_z(x, lamb, s, G, g, C, d)
        delta = np.linalg.solve(Mat,b)

        # We use the step-size correction function previously defined

        alpha = Newton_step(lamb, delta[n:n+m], s, delta[n+m:])

        # We compute the correction param.

        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[n+m:]).dot(lamb + alpha * delta[n:(n + m)])) / m
        sigma = (Mu/mu)**3

        # We correct the sub-step

        b[(n + m):] = b[(n + m):] - np.diag(delta[(n + m):]*delta[n:(n + m)]).dot(e) + sigma * mu * e

        # We find delta by using the linalg function

        delta = np.linalg.solve(Mat, b)

        #  Step-size correction with delta

        alpha = Newton_step(lamb, delta[n:(n + m)], s, delta[(n + m):])

        # Let's update the sub-step

        z = z + (alpha * delta) * 0.95

        # Stopping citeria

        if (np.linalg.norm(-b[:n]) < epsilon) or (np.linalg.norm(-b[n:(n + m)]) < epsilon) or (np.abs(mu) < epsilon):

            break

        # We update the Matrix_KKT

        x = z[:n]
        lamb = z[n:(n + m)]
        s = z[(n + m):]
        Mat, S, Lambdas = Matrix_KKT(G, C, n, m, lamb, s)

    # We print all the results obtained

    if Print_Time == "Yes":

        End = time.time()

        if Print_Results == "Yes":

            print("The computational time for the test problem is equal to: ", End - Start)

    if Print_Results == "Yes":

        print('The minimum of the function was found:', F(x, G, g))
        print('The real minimum is:', F(-g, G, g))
        print('Iterations needed:', i)
        print('Condition number:', np.linalg.cond(Mat))

    return(End - Start, i, abs(F(x, G, g) - F(-g, G, g)), np.linalg.cond(Mat))

# We can call the function like this:
result = Func_C3(n=10, Print_Time="Yes", Print_Results="Yes")


#C4
#We will modify the program of C2 and C3.

#C4.Strategy1
#Firstly, we will work with the strategy 1:#

# We create the KKT Matrix, lambda and S matrices.

def M_KKT_1(G, C, lamb, s):

    S = np.diag(s)
    Lambdas = np.diag(lamb)
    eq1 = np.concatenate((G, -C),axis = 1)
    eq2 = np.concatenate((- np.transpose(C), - np.diag(1 / lamb * s)), axis = 1)
    Mat = np.concatenate((eq1, eq2))

    return Mat, S, Lambdas


def Func_C4_strategy_1(n, maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "No"):
    np.random.seed(2) # Set the seed for random number generation
    random.seed(2)  # Set the seed for the random module

    # We define all the parameters we will need

    m = 2*n
    x = np.zeros((n))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x,lamb,s))
    G = np.identity(n)
    C = np.concatenate((G,-G),axis = 1)
    d = np.full((m), -10)
    e = np.ones((m))
    g = np.random.normal(0, 1, (n))

    # We count the time if asked for when calling the function

    if Print_Time == "Yes":
      np.random.seed(4)
      Start = time.time()

    # Create Matrix_KKT, S and Lambdas with the previously defined function

    Mat, S, Lambda = M_KKT_1(G, C, lamb, s)

    for i in range(maxIter):

        lamb_inv = np.diag(1/lamb)

        b = F_z(x, lamb, s, G, g, C, d)
        r1 = b[:n]
        r2 = b[n:(n + m)]
        r3 = b[(n + m):]
        b = np.concatenate(([- r1, - r2 + 1/ lamb * r3]))

        # LDL factorization

        L, D, perm = ldl(Mat)
        y = solve_triangular(L, b, lower=True, unit_diagonal = True)
        delta = solve_triangular(D.dot(np.transpose(L)), y, lower = False)
        deltaS = lamb_inv.dot(- r3 - s * delta[n:])
        delta = np.concatenate((delta, deltaS))

        # We use th step-size correction function previously defined

        alpha = Newton_step(lamb, delta[n:(n + m)], s, delta[(n + m):])

        # We compute the correction param.

        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[(n + m):]).dot(lamb + alpha * delta[n:(n + m)])) / m
        sigma = (Mu / mu) ** 3

        # Corrector substep

        Ds = np.diag(delta[(n + m):] * delta[n:(n + m)])
        b = np.concatenate((-r1, -r2 + lamb_inv.dot(r3 + Ds.dot(e) - sigma * mu * e)))

        # We repeat the LDL factorization

        y = solve_triangular(L, b, lower = True, unit_diagonal = True)
        delta = solve_triangular(D.dot(np.transpose(L)), y, lower = False)
        deltaS = lamb_inv.dot(-r3 - Ds.dot(e) + sigma * mu * e - s * delta[n:])
        delta = np.concatenate((delta, deltaS))

        # Step-size correction substep

        alpha = Newton_step(lamb, delta[n:(n + m)], s, delta[(n + m):])

        # Update substep

        z = z + (alpha * delta) * 0.95

        # Stopping citeria

        if (np.linalg.norm(- b[:n]) < epsilon) or (np.linalg.norm(- b[n:(n + m)]) < epsilon) or (np.abs(mu) < epsilon):
            break

        # We update the Matrix_KKT

        x = z[:n]
        lamb = z[n:(n + m)]
        s = z[(n + m):]
        Mat, S, Lambda = M_KKT_1(G, C, lamb, s)

    # We print all the results obtained

    if Print_Time == "Yes":

        End = time.time()

        if Print_Results == "Yes":

            print("The computational time for the test problem is equal to: ", End - Start)

    if Print_Results == "Yes":

        print('The minimum of the function was found:', F(x, G, g))
        print('The real minimum is:', F(-g, G, g))
        print('Iterations needed:', i)
        print('Condition number:', np.linalg.cond(Mat))

    return(End - Start, i, abs(F(x, G, g) - F(-g, G, g)), np.linalg.cond(Mat))



#C4.Strategy2
#Now, let's work with the strategy 2:
# We create the KKT Matrix, lambda and S matrices.

def M_KKT_2(G, C, lamb, s):

    S = np.diag(s)
    Lambdas = np.diag(lamb)
    Mat = G + C.dot(np.diag(1 / s * lamb)).dot(np.transpose(C))

    return Mat, Lambdas, S

def Func_C4_strategy_2(n, maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "No"):
    np.random.seed(2) # Set the seed for random number generation
    random.seed(2)  # Set the seed for the random module

    # We define all the parameters we will need

    m = 2 * n
    x = np.zeros((n))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x, lamb, s))
    G = np.identity(n)
    C = np.concatenate((G, - G),axis = 1)
    d = np.full((m), - 10)
    e = np.ones((m))
    g = np.random.normal(0, 1, (n))

    # We count the time if asked for when calling the function

    if Print_Time == "Yes":
        np.random.seed(4)
        Start = time.time()

    # Create Matrix_KKT, S and Lambdas with the previously defined function

    Ghat, Lambda, S  = M_KKT_2(G, C, lamb,s)

    for i in range(maxIter):

        S_inv = np.diag(1 / s)

        b = F_z(x, lamb, s, G, g, C, d)
        r1 = b[:n]
        r2 = b[n:(n + m)]
        r3 = b[(n + m):]
        rhat = - C.dot(np.diag(1 / s)).dot((- r3 + lamb * r2))
        b = - r1 - rhat

        # Cholesky factorization

        Cholesk = cholesky(Ghat, lower = True)
        y = solve_triangular(Cholesk, b, lower=True)
        delta_x = solve_triangular(np.transpose(Cholesk), y)
        delta_lamb = S_inv.dot((- r3 + lamb * r2)) - S_inv.dot(Lambda.dot(np.transpose(C)).dot(delta_x))
        delta_s = - r2 + np.transpose(C).dot(delta_x)
        delta = np.concatenate((delta_x,delta_lamb, delta_s))

        # We use th step-size correction function previously defined

        alpha = Newton_step(lamb, delta[n:(n + m)], s, delta[(n + m):])

        # We compute the correction param.

        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[(n + m):]).dot(lamb + alpha * delta[n:(n + m)])) / m
        sigma = (Mu / mu) ** 3

        # Corrector substep

        Ds_Dlamb = np.diag(delta[n+m:]*delta[n:n+m])
        b = -r1-(-C.dot(np.diag(1/s)).dot((-r3-Ds_Dlamb.dot(e)+sigma*mu*e+lamb*r2)))

        # We repeat the Cholesky factorization again

        y = solve_triangular(Cholesk,b,lower=True)
        delta_x = solve_triangular(np.transpose(Cholesk),y)
        delta_lamb = S_inv.dot(-r3-Ds_Dlamb.dot(e)+sigma*mu*e+lamb*r2)-S_inv.dot(lamb*(np.transpose(C).dot(delta_x)))
        delta_s = - r2 + np.transpose(C).dot(delta_x)
        delta = np.concatenate((delta_x,delta_lamb, delta_s))

        # Step-size correction substep

        alpha = Newton_step(lamb, delta[n:(n + m)],s,delta[(n + m):])

        # Update substep

        z = z + (alpha * delta) * 0.95

        # Stopping citeria

        if (np.linalg.norm(- r1) < epsilon) or (np.linalg.norm(-r2) < epsilon) or (np.abs(mu) < epsilon):

            break

        # We update the Matrix_KKT

        x = z[:n]
        lamb = z[n:(n + m)]
        s = z[(n + m):]
        Ghat, Lambda, S = M_KKT_2(G, C, lamb,s)

    if Print_Time == "Yes":

        End = time.time()

        if Print_Results == "Yes":

            print("Computation time for the test problem: ", End - Start)

    if Print_Results == "Yes":

        print('The minimum of the function was found:', F(x, G, g))
        print('The real minimum is:', F(-g, G, g))
        print('Iterations needed:', i)
        print('Condition number:', np.linalg.cond(Ghat))

    return(End - Start, i, abs(F(x, G, g) - F(-g, G, g)), np.linalg.cond(Ghat))


#We will compare the results of each function/program (C3, C4_LDL, C4_Cholesky) for different values of n (10, 50, 100):

#n=10
print("C3")
Func_C3(n=10, Print_Time="Yes", Print_Results="Yes")
print("\n")

print("C4_LDL")
Func_C4_strategy_1(n=10, maxIter=100, epsilon=10e-16, Print_Time="Yes", Print_Results="Yes")
print("\n")

print("C4_Cholesky")
Func_C4_strategy_2(n=10, maxIter=100, epsilon=10e-16, Print_Time="Yes", Print_Results="Yes")


#n=50
print("C3")
Func_C3(n=50, Print_Time="Yes", Print_Results="Yes")
print("\n")

print("C4_LDL")
Func_C4_strategy_1(n=50, maxIter=100, epsilon=10e-16, Print_Time="Yes", Print_Results="Yes")
print("\n")

print("C4_Cholesky")
Func_C4_strategy_2(n=50, maxIter=100, epsilon=10e-16, Print_Time="Yes", Print_Results="Yes")

#n=100
print("C3")
Func_C3(n=100, Print_Time="Yes", Print_Results="Yes")
print("\n")

print("C4_LDL")
Func_C4_strategy_1(n=100, maxIter=100, epsilon=10e-16, Print_Time="Yes", Print_Results="Yes")
print("\n")

print("C4_Cholesky")
Func_C4_strategy_2(n=100, maxIter=100, epsilon=10e-16, Print_Time="Yes", Print_Results="Yes")



#C5

def ReadMatrix(source, shape, symm=False):

    matrix = np.zeros(shape)

    with open(source, "r") as file:

        a = file.readlines()

    for line in a:

        row, column, value = line.strip().split()
        row = int(row)
        column = int(column)
        value = float(value)
        matrix[row - 1, column - 1] = value

        if symm == True:

            matrix[column - 1, row - 1] = value

    return matrix


def ReadVector(source, n):

    v = np.zeros(n)

    with open(source, "r") as file:

        a = file.readlines()

    for line in a:

        idx, value = line.strip().split()
        idx = int(idx)
        value = float(value)
        v[idx - 1] = value

    return v


# We define the Matrix KKT for the exercise

def M_KKT_C5(G, C, A, n, m, p, lamb,s):

    S = np.diag(s)
    Lambda = np.diag(lamb)
    temp1 = np.concatenate((G, -A, -C, np.zeros((n, m))),axis = 1)
    temp2 = np.concatenate((- np.transpose(A),np.zeros((p, p + 2 * m))), axis = 1)
    temp3 = np.concatenate((np.transpose(- C),np.zeros((m, p + m)), np.identity(m)), axis = 1)
    temp4 = np.concatenate((np.zeros((m, n + p)), S, Lambda), axis = 1)
    M = np.concatenate((temp1, temp2, temp3, temp4))

    return M, S, Lambda



def funC5(A, G, C, g, x, gamma, lamb, s, bm, d):

    comp1 = G.dot(x)+g-A.dot(gamma)-C.dot(lamb)
    comp2 = bm-np.transpose(A).dot(x)
    comp3 = s+d-np.transpose(C).dot(x)
    comp4 = s*lamb

    return np.concatenate((comp1,comp2,comp3,comp4))


def Function_C5(maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "No", Data = r"optpr1"):
    
    np.random.seed(2) # Set the seed for random number generation
    random.seed(2)  # Set the seed for the random module

    # We define all the parameters we will need

    n = int(np.loadtxt(Data + "/g.dad")[:,0][-1])
    p = n // 2
    m = 2 * n
    A = ReadMatrix(Data + "/A.dad", (n, p))
    bm = ReadVector(Data + "/b.dad", p)
    C = ReadMatrix(Data + "/C.dad", (n, m))
    d = ReadVector(Data + "/d.dad", m)
    e = np.ones((m))
    G = ReadMatrix(Data + "/g.dad", (n, n), True)
    g = np.zeros(n)
    x = np.zeros((n))
    gamma = np.ones((p))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x,gamma,lamb,s))

    if Print_Time == "Yes":
        np.random.seed(2)
        Start = time.time()

    # Create Matrix_KKT, S and Lambdas with the previously defined function

    Mat, S, Lambda = M_KKT_C5(G, C, A, n, m, p, lamb, s)

    for i in range(maxIter):

        b = - funC5(A, G, C, g, x, gamma, lamb, s, bm, d)
        delta = np.linalg.solve(Mat, b)

        # Step-size correction substep

        alpha = Newton_step(lamb,delta[(n + p) : (n + p + m)], s, delta[(n + m + p):])

        # Compute correction parameters

        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[(n + m + p):]).dot(lamb + alpha * delta[(n + p):(n + m + p)])) / m
        sigma = (Mu / mu) ** 3

        # Corrector substep

        b[(n + m + p):] = b[(n + p + m):] - np.diag(delta[(n + p + m):] * delta[(n + p) : (n + p + m)]).dot(e) + sigma * mu * e
        delta = np.linalg.solve(Mat, b)

        # Step-size correction substep

        alpha = Newton_step(lamb, delta[(n + p):(n + p + m)], s, delta[(n + m + p):])

        # We update the substep

        z = z + 0.95 * alpha * delta

        # The stopping criteria

        if (np.linalg.norm(- b[:n]) < epsilon) or (np.linalg.norm(- b[n:(n + m)]) < epsilon) or (np.linalg.norm(- b[(n + p):(n + p + m)]) < epsilon) or (np.abs(mu) < epsilon):

            break

        # We update the Matrix KKT

        x = z[:n]
        gamma = z[n:(n+p)]
        lamb = z[(n + p):(n + m + p)]
        s = z[(n + m + p):]
        Mat, S, Lambda = M_KKT_C5(G, C, A, n, m, p, lamb,s)

    condition_number = np.linalg.cond(Mat)

    if Print_Time == "Yes":

        End = time.time()

        if Print_Results == "Yes":

            print("Computation time: ",End - Start)

    if Print_Results == "Yes":

        print('Minimum was found:', F(x, G, g))
        print('Condition number:', condition_number)
        print('Iterations needed:', i)


print("For matrices and vectors from optpr1, the obtained results are the following:")
print("\n")
Function_C5(maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "Yes", Data = r"optpr1")

print("\n")

print("For matrices and vectors from optpr2, the obtained results are the following:")
print("\n")
Function_C5(maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "Yes", Data = r"optpr2")


#C6

# We define the Matrix KKT for the exercise

def M_KKT_C6(G, C, A, n, m, p, lamb,s):
    
    S = np.diag(s)
    Lambda = np.diag(lamb)
    temp1 = np.concatenate((G,- A, - C),axis = 1)
    temp2 = np.concatenate((- np.transpose(A), np.zeros((p, p + m))), axis = 1)
    temp3 = np.concatenate((- np.transpose(C), np.zeros((m, p)), np.diag(-1 / lamb * s)), axis = 1)
    Mat = np.concatenate((temp1, temp2, temp3))
    
    return Mat, S, Lambda


def Function_C6(maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "No", Data = "optpr1"):
    np.random.seed(2) # Set the seed for random number generation
    random.seed(2)  # Set the seed for the random module

    # We define all the parameters we will need
    
    n = int(np.loadtxt(Data + "/G.dad")[:,0][-1])
    p = n // 2
    m = 2 * n
    A = ReadMatrix(Data + "/A.dad", (n, p))
    bm = ReadVector(Data + "/b.dad", p)
    C = ReadMatrix(Data + "/C.dad", (n, m))
    d = ReadVector(Data + "/d.dad", m)
    e = np.ones((m))
    G = ReadMatrix(Data + "/G.dad", (n, n), True)
    g = np.zeros((n))
    x = np.zeros((n))
    gamma = np.ones((p))
    lamb = np.ones((m))
    s = np.ones((m))
    z = np.concatenate((x, gamma, lamb, s))
    
    if Print_Time == "Yes":
        np.random.seed(2)
        Start = time.time()

    # Create Matrix_KKT, S and Lambdas with the previously defined function 
    
    Mat,S,Lamb = M_KKT_C6(G, C, A, n, m, p, lamb,s)

    for i in range(maxIter):
        
        lamb_inv = np.diag(1/lamb)

        b = funC5(A, G, C, g, x, gamma, lamb, s, bm, d)
        r1, r2, r3, r4 = b[:n], b[n:n+p], b[n+p:n+p+m], b[n+p+m:]
        b = np.concatenate(([-r1,-r2,-r3+1/lamb*r4]))

        # LDL factorization
        
        L, D, perm = ldl(Mat)
        y = np.linalg.solve(L, b)
        delta = np.linalg.solve(D.dot(np.transpose(L)), y)
        deltaS = lamb_inv.dot(- r4 - s * delta[(n + p):])
        delta = np.concatenate((delta, deltaS))

        # Step-size correction substep
        
        alpha = Newton_step(lamb,delta[(n + p) : (n + p + m)], s, delta[(n + m + p):])

        # We compute the correction parameters
        
        mu = s.dot(lamb) / m
        Mu = ((s + alpha * delta[(n + m + p):]).dot(lamb + alpha * delta[(n + p):(n + m + p)])) / m
        sigma = (Mu / mu) ** 3

        # Substep corrector
        
        Ds = np.diag(delta[(n + p + m):] * delta[(n + p):(n + p + m)])
        b = np.concatenate((- r1, - r2, - r3 + lamb_inv.dot(r4 + Ds.dot(e) - sigma * mu * e)))

        # Repeat LDL factorization
        
        y = np.linalg.solve(L, b)
        delta = np.linalg.solve(D.dot(np.transpose(L)), y)
        deltaS = lamb_inv.dot(- r4 - Ds.dot(e) + sigma * mu * e - s * delta[(n + p):])
        delta = np.concatenate((delta, deltaS))

        # Step-size correction substep
        
        alpha = Newton_step(lamb, delta[(n + p):(n + p + m)], s, delta[(n + m + p):])

        # We update the substep
        
        z = z + 0.95 * alpha * delta

        # The stopping criteria
        
        if (np.linalg.norm(- b[:n]) < epsilon) or (np.linalg.norm(- b[n:(n + m)]) < epsilon) or (np.linalg.norm(- b[(n + p):(n + p + m)]) < epsilon) or (np.abs(mu) < epsilon):
            
            break

        # We update tha  Matrix KKT
        
        x = z[:n]
        gamma = z[n:(n+p)]
        lamb = z[(n + p):(n + m + p)]
        s = z[(n + m + p):]
        Mat,Lamb,S = M_KKT_C6(G, C, A, n, m, p, lamb,s)
        
    condition_number = np.linalg.cond(Mat)
    
    if Print_Time == "Yes":
        
        End = time.time()
        
        if Print_Results == "Yes":
            
            print("Computation time: ",End - Start)
            
    if Print_Results == "Yes":
        
        print('Minimum was found:', F(x, G, g))
        print('Condition number:', condition_number)
        print('Iterations needed:', i)


print("For matrices and vectors from optpr1, the obtained results where the following:") 
print("\n")
Function_C6(maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "Yes", Data = "optpr1")

print("\n")

print("For matrices and vectors from optpr2, the obtained results where the following:")
print("\n")
Function_C6(maxIter=100, epsilon=10e-16, Print_Time = "Yes", Print_Results = "Yes", Data = "optpr2")
