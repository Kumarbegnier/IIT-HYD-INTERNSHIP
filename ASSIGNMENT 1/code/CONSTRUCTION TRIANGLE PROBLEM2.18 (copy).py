import numpy as np
import math 
import matplotlib.pyplot as plt
from coeffs import *

#if using termux
#import subprocess
#import shlex
#end if

#sides
c = 6 

#FOR TRINAGLE 
angleX= 30
angleY = 100
angleZ = 180-(angleX+angleY)
print(angleZ)

#WE ASSUME ONE "D" POINT WHICH IS PERPENDICULAR TO THE AC LINE
q = 6*(math.sin(math.radians(30)))
p = 6*(math.cos(math.radians(30)))
print(q,p)




# print dc yused in triangle
dz= 3/(math.tan(math.radians(50)))
print(dz)


#for finding the coordinate of C[b,0]
b=(p+dz)
print(b)

#Triangle vertices
X = np.array([0,0]) 
Y = np.array([p,q]) 
Z = np.array([b,0]) 

#Generating all lines
x_XY = line_gen(X,Y)
x_YZ = line_gen(Y,Z)
x_ZX = line_gen(Z,X)

#Plotting all lines
plt.plot(x_XY[0,:],x_XY[1,:],label='$XY$')
plt.plot(x_YZ[0,:],x_YZ[1,:],label='$YZ$')
plt.plot(x_ZX[0,:],x_ZX[1,:],label='$ZX$')

plt.plot(X[0], X[1], 'o')
plt.text(X[0] * (1 + 0.1), X[1] * (1 - 0.1) , 'X')
plt.plot(Y[0], Y[1], 'o')
plt.text(Y[0] * (1 - 0.2), Y[1] * (1) , 'Y')
plt.plot(Z[0], Z[1], 'o')
plt.text(Z[0] * (1 + 0.03), Z[1] * (1 - 0.1) , 'Z')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor

#if using termux
#plt.savefig('../figs/rt_triangle.pdf')
#plt.savefig('../figs/rt_triangle.eps')
#subprocess.run(shlex.split("termux-open ../figs/rt_triangle.pdf"))
#else
plt.show()







