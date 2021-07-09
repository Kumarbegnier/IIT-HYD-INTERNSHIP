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
angleA= 30
angleB = 100
angleC = 180-(angleA+angleB)
print(angleC)

#WE ASSUME ONE "D" POINT WHICH IS PERPENDICULAR TO THE AC LINE
q = 6*(math.sin(math.radians(30)))
p = 6*(math.cos(math.radians(30)))
print(q,p)




# print dc yused in triangle
dc= 3/(math.tan(math.radians(50)))
print(dc)


#for finding the coordinate of C[b,0]
b=(p+dc)
print(b)

#Triangle vertices
A = np.array([0,0]) 
B = np.array([p,q]) 
C = np.array([b,0]) 

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')

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







