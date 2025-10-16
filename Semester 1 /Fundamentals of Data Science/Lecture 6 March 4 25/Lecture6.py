# a) Given function y(x) = x e -x, show that its derivative is y’(x) = (1-x) e -x;
# b) Generate an evenly spaced array X in the interval [0:4] with the step of 0.04;
# c) Generate corresponding array Y, where Yi=Xi exp(-Xi) (i.e. the function from
# 1a),
# d) Calculate corresponding array DY1, where DYi=(1-Xi ) exp(-Xi), i.e. derivative
# of the function from 1a,
# e) Calculate derivative DY2 of the tabulated function Yi(Xi) defined in 1c. Use
# CD2 scheme (see lecture notes, Yi’ = (Yi+1 - Yi-1)/(Xi+1 - Xi-1) ). (NB Obviously,
# your array DY2 would be 2 elements shorter than arrays X and Y)
# f) Compare DY1 and DY2, i.e. derivatives of y(x) = x e -x, calculated analytically
# and numerically.


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * np.exp(-x)

def df(x):
    return (1-x) * np.exp(-x)

h = 0.004 

x = np.arange(0, 4, h)

y = f(x)

dy1 = df(x)

dy2 = np.zeros(len(x)-2)


for i in range(1, len(x)-1):
    dy2[i-1] = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])

plt.plot(x, dy1, label='Analytical')
plt.plot(x[1:-1], dy2, label='Numerical')
plt.legend()
plt.show()