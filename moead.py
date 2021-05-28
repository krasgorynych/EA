# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:11:57 2021

@author: Олексий
"""

import operator
import functools
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations

def Tchebycheff(individual,weight,z):
    temp = []
    for i in range(len(individual.obj)):
        temp.append(weight[i]*np.abs(individual.obj[i]-z[i]))

    return np.max(temp)

def crossMutation(pop_parent,  D, lower, upper, pc, pm ,problem, yita1= 20, yita2 = 20):
    pop_offspring = []
    for i in range(round(len(pop_parent) / 2)):
        parent_1 = round(len(pop_parent) * random.random())
        if (parent_1 == len(pop_parent)):
            parent_1 = len(pop_parent) - 1
        parent_2 = round(len(pop_parent) * random.random())
        if (parent_2 == len(pop_parent)):
            parent_2 = len(pop_parent) - 1
        while (parent_1 == parent_2):
            parent_1 = round(len(pop_parent) * random.random())
            if (parent_1 == len(pop_parent)):
                parent_1 = len(pop_parent) - 1
        parent1 = pop_parent[parent_1]
        parent2 = pop_parent[parent_2]
        off1 = parent1
        off2 = parent2
        if (random.random() < pc):
            off1x = []
            off2x = []
            for j in range(D):
                if random.random() <= 0.5:
                    if (np.fabs(parent1.x[j] - parent2.x[j]) > 1e-9):
                        if (parent1.x[j] < parent2.x[j]):
                            y1 = parent1.x[j]
                            y2 = parent2.x[j]
                        else:
                            y1 = parent2.x[j]
                            y2 = parent1.x[j]
                        rand = random.random()
                        yl = lower[j]
                        yu = upper[j]
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(yita1 + 1.0))
                        if (rand <= (1.0 / alpha)):
                            betaq = pow((rand * alpha), (1.0 / (yita1 + 1.0)))
                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (yita1 + 1.0)))

                        off11 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(yita1 + 1.0))
                        if (rand <= (1.0 / alpha)):
                            betaq = pow((rand * alpha), (1.0 / (yita1 + 1.0)))
                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (yita1 + 1.0)))
                        off22 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                        if (off11 > upper[j]):
                            off11 = upper[j]
                        elif (off11 < lower[j]):
                            off11 = lower[j]
                        if (off22 > upper[j]):
                            off22 = upper[j]
                        elif (off22 < lower[j]):
                            off22 = lower[j]
                        if random.random() <= 0.5:
                            off1x.append(off22)
                            off2x.append(off11)
                        else:
                            off1x.append(off11)
                            off2x.append(off22)
                    else:
                        off1x.append(parent1.x[j])
                        off2x.append(parent2.x[j])
                else:
                    off1x.append(parent1.x[j])
                    off2x.append(parent2.x[j])
            off1 = Individual(off1x,problem)
            off2 = Individual(off2x,problem)

        off1x = []
        off2x = []
        for j in range(D):
            if (random.random() < pm):
                y1 = off1.x[j]
                y2 = off2.x[j]
                ylow = lower[j]
                yu = upper[j]
                delta11 = (y1 - ylow) / (yu - ylow)
                delta12 = (yu - y1) / (yu - ylow)
                delta21 = (y2 - ylow) / (yu - ylow)
                delta22 = (yu - y2) / (yu - ylow)
                rnd = random.random()
                mut_pow = 1.0 / (yita2 + 1.0)
                if (rnd <= 0.5):
                    xy1 = 1.0 - delta11;
                    xy2 = 1.0 - delta21;
                    val1 = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy1, (yita2 + 1.0)));
                    val2 = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy2, (yita2 + 1.0)));
                    deltaq1 = pow(val1, mut_pow) - 1.0
                    deltaq2 = pow(val2, mut_pow) - 1.0
                else:
                    xy1 = 1.0 - delta12;
                    xy2 = 1.0 - delta22;
                    val1 = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy1, (yita2 + 1.0)));
                    val2 = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy2, (yita2 + 1.0)));

                    deltaq1 = 1.0 - (pow(val1, mut_pow));
                    deltaq2 = 1.0 - (pow(val2, mut_pow))

                off11 = float(off1.x[j] + deltaq1 * (upper[j] - lower[j]))
                off22 = float(off2.x[j] + deltaq2 * (upper[j] - lower[j]))
                if (off11 > upper[j]):
                    off11 = upper[j]
                elif (off11 < lower[j]):
                    off11 = lower[j]
                if (off22 > upper[j]):
                    off22 = upper[j]
                elif (off22 < lower[j]):
                    off22 = lower[j]
                off1x.append(off11)
                off2x.append(off22)
            else:
                off1x.append(off1.x[j])
                off2x.append(off2.x[j])
        off1 = Individual(off1x,problem)
        off2 = Individual(off2x,problem)
        pop_offspring.append(off1)
        pop_offspring.append(off2)
    return pop_offspring

def HV(pop,flag = None):

    temp = []
    for i in range(len(pop)):
        temp.append(pop[i].obj)

    popObj = np.vstack(temp)


    N, M = np.shape(popObj)
    if flag == None:
        refPoint = np.max(popObj, 0)
    else:
        refPoint = np.max(popObj * 1.1, 0)
    if len(popObj) == 0:
        score = 0
    else:
        pl = sorted(popObj, key=lambda x: x[0])
        pl = np.array(pl)
        temp = [1]
        temp.append(pl)
        S = [temp]
        for k in range(M - 1):
            S_ = []
            for i in range(len(S)):
                Stemp = Slice(S[i][1], k, refPoint)
                for j in range(len(Stemp)):
                    temp = []
                    temp.append(Stemp[j][0] * S[i][0])  # depth
                    temp.append(Stemp[j][1])
                    S_.append(temp)
            S = S_
        score = 0
        for i in range(len(S)):
            p = Head(S[i][1])
            score = score + S[i][0] * np.abs(p[M - 1] - refPoint[M - 1])
    return score


def Slice(pl, k, refPoint):
    p = Head(pl)
    pl = Tail(pl)
    ql = []
    S = []
    while len(pl) != 0:
        temp = []
        ql = Insert(p, k + 1, ql)
        p_ = Head(pl)
        temp.append(np.abs(p[k] - p_[k]))  # depth
        temp.append(ql)
        S.append(temp)
        p = p_
        pl = Tail(pl)

    temp = []
    ql = Insert(p, k + 1, ql)
    temp.append(np.abs(p[k] - refPoint[k]))
    temp.append(ql)
    S.append(temp)

    return S


def Insert(p, k, pl):
    flag1 = 0
    flag2 = 0
    ql = []
    hp = Head(pl)
    while len(pl) != 0 and hp[k] < p[k]:
        ql.append(hp)
        pl = Tail(pl)
        hp = Head(pl)
    ql.append(p)
    m = len(p)
    while len(pl) != 0:
        q = Head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            elif p[i] > q[i]:
                flag2 = 1

        if ~(flag1 == 1 and flag2 == 0):
            ql.append(Head(pl))
        pl = Tail(pl)

    return ql


def Head(pl):
    if len(pl) == 0:
        p = []
    else:
        p = pl[0]
    return p



def Tail(pl):
    if len(pl) < 2:
        ql = []
    else:
        ql = pl[1:]
    return ql

def checkDominance(individual1,individual2):
    flag1 = 0
    flag2 = 0
    for i in range(individual1.problem.M):

        if(individual1.obj[i] < individual2.obj[i]):
            flag1 = 1
        elif(individual2.obj[i] < individual1.obj[i]):
            flag2 = 1
    if(flag1 == 1 and flag2 == 0):

        return 1
    elif(flag1 == 0 and flag2 == 1):

        return -1
    else:
        return 0

class Individual():

    def __init__(self, x, problem):
        self.x = x
        self.problem = problem
        self.obj = self.problem.evaluate(x)
        self.fitness = 0
        self.cd = 0
        self.paretoRank = 0

def factorial(n):
    result = 1
    for i in range(2,n+1):
        result*=i
    return result


def comb(n,k):
    return factorial(n)/(factorial(n-k)*factorial(k))

def UniformPoint(N,M):
    H1 = 1;
    while(comb(H1+M, M-1) <= N):
        H1 = H1 + 1

    temp1 = list(combinations(np.arange(H1+M-1), M-1))
    temp1 = np.array(temp1)
    temp2 = np.arange(M-1)
    temp2 = np.tile(temp2, (int(comb(H1+M-1, M-1)), 1))
    W = temp1-temp2
    W = (np.concatenate((W, np.zeros((np.size(W,0),1))+H1), axis=1)-np.concatenate((np.zeros((np.size(W,0),1)), W), axis = 1))/H1

    if H1<M:
        H2 = 0
        while(comb(H1+M-1,M-1)+comb(H2+M,M-1)<=N):
            H2 = H2 + 1
        if H2>0:
            temp1 = list(combinations(np.arange(H2 + M - 1), M - 1))
            temp1 = np.array(temp1)
            temp2 = np.arange(M - 1)
            temp2 = np.tile(temp2, (int(comb(H2 + M - 1, M - 1)), 1))
            W2 = temp1 - temp2
            W2 = (np.concatenate((W2, np.zeros((np.size(W2,0),1))+H2), axis=1)-np.concatenate((np.zeros((np.size(W2,0),1)), W2), axis = 1))/H2
            W = np.concatenate((W,W2/2+1/(2*M)),axis=0)

    realN = np.size(W,0)
    W[W==0] = 10**(-6)
    return W,realN

def initial(N,D,M,lower,upper,problem,encoding):

    if encoding == 'real':
        p = []
        for i in range(N):
            tempDecVar = []
            for j in range(D):
                tempX = lower[j]+(upper[j]-lower[j])*random.random()
                tempDecVar.append(tempX)
            p.append(Individual(tempDecVar, problem))
        return p
    elif encoding == 'binary':
        #  To do.....


        pass
    elif encoding == 'permutation':
        # To do.....


        pass

def Best(P):
    best = []
    for i in range(len(P[0].obj)):
        best.append(P[0].obj[i])
    for i in range(len(P)):
        for j in range(len(P[i].obj)):
            if P[i].obj[j] < best[j]:
                best[j] = P[i].obj[j]
    return best

def FindNeighbour(W,N,M,T):
    B = []
    for i in range(N):
        temp = []
        for j in range(N):
            distance = 0
            for k in range(M):
                distance+=(W[i][k]-W[j][k])**2
            distance = np.sqrt(distance)
            temp.append(distance)
        index = np.argsort(temp)
        B.append(index[:T])
    return B

def MOEAD(N,maxgen,problem,encoding,type = 1):
    start = time.time()


    D = problem.D
    M = problem.M
    lower = problem.lower
    upper = problem.upper
    pc = 1
    pm = 1 / D

    W, N = UniformPoint(N, M)
    T = np.ceil(N/10)
    T = int(T)
    B = FindNeighbour(W, N, M, T)

    pop = initial(N, D, M, lower, upper, problem, encoding)
    z = Best(pop)

    gen = 1
    plt.ion()
    fig = plt.figure()

    while gen<=maxgen:
        for i in range(N):
            k = np.random.randint(0,T)
            l = np.random.randint(0,T)
            while k==l:
                l = np.random.randint(0,T)
            pop_parent = [pop[B[i][k]],pop[B[i][l]]]
            pop_offspring = crossMutation(pop_parent, D, lower, upper, pc, pm, problem)
            if checkDominance(pop_offspring[0],pop_offspring[1]):
                y = pop_offspring[0]
            else:
                y = pop_offspring[1]

            for j in range(len(z)):
                if y.obj[j]<z[j]:
                    z[j] = y.obj[j]

            for j in range(T):
                if Tchebycheff(y,W[B[i][j]],z) < Tchebycheff(pop[B[i][j]],W[B[i][j]],z):
                    pop[B[i][j]] = y


        draw(problem,pop, M, fig)
        # print(Tchebycheff(pop[99],W[99],z))
        # print(pop[99].obj[0],pop[99].obj[1])
        if gen < maxgen:
            plt.clf()

        if (gen % 10) == 0:
            print("%d gen has completed!\n" % gen)
        gen = gen + 1;
    end = time.time()
    plt.ioff()
    print("runtime：%2fs" % (end - start))
    # for i in range(N):
    #     print("population %f obj[0]: %f obj[1]: %f obj[2]: %f"%(i,pop[i].obj[0],pop[i].obj[1],pop[i].obj[2]))
    #     print("x[0]:%f x[1]:%f x[2]:%f x[3]:%f x[4]:%f x[5]:%f  x[6]:%f x[7]:%f x[8]:%f"
    #           %(pop[i].x[0],pop[i].x[1],pop[i].x[2],pop[i].x[3],pop[i].x[4],pop[i].x[5],pop[i].x[6],pop[i].x[7],pop[i].x[8])
    #           )
    #     print("here")

    # PRINT INDICATOR

    score = HV(pop)

    print("HV indicator:%f" % score)


from mpl_toolkits.mplot3d import Axes3D


def draw(problem,pop, M,  fig = None, ax = None):

    x = []
    y = []
    z = []
    if M == 2:
        for i in range(len(pop)):
            x.append(pop[i].obj[0])
            y.append(pop[i].obj[1])
        
        plt.scatter(x, y, marker='o', color='#0139DD', s=17)

        plt.xlabel('f1')
        plt.ylabel('f2')
        #plt.show()
        plt.draw()
        plt.pause(0.003)
    elif M == 3:
        for i in range(len(pop)):
            x.append(pop[i].obj[0])
            y.append(pop[i].obj[1])
            z.append(pop[i].obj[2])
    
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c='#0139DD',s=22)
        ax.view_init(elev=40, azim=40)
        plt.draw()
        plt.pause(0.003)
    elif M > 3:
        label = list(range(1, M+1))
        for i in range(len(pop)):
            plt.plot(label, pop[i].obj, c='#3B6575', linewidth='1.2')

        plt.draw()
        plt.pause(0.003)


    plt.show()



class DTLZ2():
    def __init__(self,D,M):
        self.D = D
        self.M = M
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):
        k = self.D - self.M + 1
        g = 0
        for i in range(k):
            g += (x[self.M-1+i]-0.5)**2
        obj = [1+g]*self.M

        for i in range(self.M):
            obj[i]*=functools.reduce(operator.mul,
                                     [np.cos(x*np.pi/2) for x in x[:self.M-i-1]],1
                                     )
            if i>0:
                obj[i]*=np.sin(x[self.M-i-1]*np.pi*0.5)

        return obj


class DTLZ3():
    def __init__(self,D,M):
        self.D = D
        self.M = M
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self, x):
        k = self.D - self.M + 1
        sum1 = 0
        for i in range(k):
            sum1 += (x[self.M-1+i]-0.5)**2 - np.cos(20*np.pi*(x[self.M-1+i]-0.5))
        g = 100 * (k + sum1)
        obj = [1 + g] * self.M
        for i in range(self.M):
            obj[i] *= functools.reduce(operator.mul,
                                       [np.cos(x * np.pi / 2) for x in x[:self.M - i - 1]], 1
                                       )
            if i > 0:
                obj[i] *= np.sin(x[self.M - i - 1] * np.pi * 0.5)

        return obj


class DTLZ4():
    def __init__(self,D,M):
        self.D = D
        self.M = M
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):
        k = self.D - self.M + 1
        g = 0
        for i in range(k):
            g += (x[self.M - 1 + i] - 0.5) ** 2
        obj = [1 + g] * self.M

        for i in range(self.M):
            obj[i] *=functools.reduce(operator.mul,
                                      [np.cos(0.5*np.pi*(x**100)) for x in x[:self.M-1-i]],1
                                      )
            if i>0:
                obj[i]*=np.sin((x[self.M-i-1])**100 * 0.5 * np.pi)
        return obj


class DTLZ5():
    def __init__(self,D,M):
        self.D = D
        self.M = M
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)


    def evaluate(self,x):
        k = self.D -self.M + 1
        g = 0
        for i in range(k):
            g += (x[self.M - 1 + i] - 0.5) ** 2
        obj = [1 + g] * self.M


        theta = []
        for i in range(self.M):
            if i == 0:
                theta.append(x[0]*np.pi*0.5)
            else:
                temp = np.pi*(1+2*g*x[i])/(4*(1+g))
                theta.append(temp)



        for i in range(self.M):
            obj[i]*=functools.reduce(operator.mul,
                                     [np.cos(theta) for theta in theta[:self.M-1-i]],1
                                     )
            if i > 0:
                obj[i]*=np.sin(theta[self.M-1-i])


        return obj



class DTLZ7():
    def __init__(self,D,M):
        self.D = D
        self.M = M
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):

        k = self.D - self.M + 1
        obj = x[:self.M-1]
        g = 1.0 + (9.0 * sum(x[self.D - k:])) / k


class ZDT1():
    def __init__(self,D):
        self.M = 2
        self.D = D
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):
        obj1 = x[0]
        g = 1+9*sum(x[1:self.D])/(self.D-1)
        h = 1-(obj1/g)**0.5
        obj2 = g*h
        obj = [obj1, obj2]
        return obj

    def PF(self,N):
        P = np.zeros((N,self.M))
        P[:, 0] = np.arange(0,1+1/N,1/(N-1))
        P[:, 1] = 1-P[:,0]**0.5
        return P




class ZDT2():
    def __init__(self,D):
        self.M = 2
        self.D = D
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):
        obj1 = x[0]
        g = 1+9*sum(x[2:self.D])/(self.D-1)
        h = 1-(obj1/g)**2
        obj2 = g*h
        obj = [obj1, obj2]
        return obj

    def PF(self,N):
        P = np.zeros((N,self.M))
        P[:, 0] = np.arange(0,1+1/N,1/(N-1))
        P[:, 1] = 1-P[:,0]**2
        return P

class ZDT3():
    def __init__(self,D):
        self.M = 2
        self.D = D
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):
        obj1 = x[0]
        g = 1+9*sum(x[1:self.D])/(self.D-1)
        h = 1-(obj1/g)**0.5-(obj1/g)*np.sin(10*np.pi*obj1)
        obj2 = g*h
        obj = [obj1,obj2]
        return obj


class ZDT4():
    def __init__(self,D):
        self.M = 2
        self.D = D
        self.lower = np.hstack([0, np.zeros(self.D-1)-5])
        self.upper = np.hstack([1, np.zeros(self.D-1)+5])

    def evaluate(self,x):
        obj1 = x[0]
        sum1 = 0
        for i in range(self.D - 1):
            sum1 = sum1 + (x[i + 1]) ** 2 - 10 * np.cos(4 * np.pi * x[i + 1])
        g =1 + 10*(self.D-1) + sum1
        h = 1-(obj1/g)**0.5
        obj2 = g*h
        obj = [obj1, obj2]
        return obj


class ZDT5():
    pass


class ZDT6():
    def __init__(self,D):
        self.M = 2
        self.D = D
        self.lower = np.zeros(self.D)
        self.upper = np.ones(self.D)

    def evaluate(self,x):
        obj1 = 1 - np.exp(-4*x[0])*(np.sin(6*np.pi*x[0])**6)
        g = 1+9*(sum(x[1:self.D])/(self.D-1))**0.25
        h = 1-(obj1/g)**2
        obj2 = g*h
        obj = [obj1,obj2]
        return obj


    
def problemSelect(problemName,D,M ):
    
    if problemName == 'ZDT2':
        problem = ZDT2(D)
    elif problemName == 'ZDT3':
        problem = ZDT3(D)
    elif problemName == 'ZDT4':
        problem = ZDT4(D)
    elif problemName == 'ZDT5':
        problem = ZDT5(D)
    elif problemName == 'ZDT6':
        problem = ZDT6(D)
    
    elif problemName == 'DTLZ2':
        if M == 0:
            problem = DTLZ2(D, 3)
        else:
            problem = DTLZ2(D, M)
    elif problemName == 'DTLZ3':
        if M == 0:
            problem = DTLZ3(D, 3)
        else:
            problem = DTLZ3(D, M)
    elif problemName == 'DTLZ4':
        if M == 0:
            problem = DTLZ4(D, 3)
        else:
            problem = DTLZ4(D, M)
    elif problemName == 'DTLZ5':
        if M == 0:
            problem = DTLZ5(D, 3)
        else:
            problem = DTLZ5(D, M)



    return problem

def PyEA(problemName,algorithm,N,maxgen,encoding,D,M=0):
    problemInstance = problemSelect(problemName,D,M)
    if algorithm == 'MOEAD':
        MOEAD(N, maxgen, problemInstance, encoding)
        
PyEA('ZDT3','MOEAD',100,300,'real',5)