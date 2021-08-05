import numpy as np 
import matplotlib.pyplot as plt 
#from coeffs import *
#import subprocess
#import shlex

l = np.array([-1,-1])
m = np.array([1,1])
O = np.array([0,0])
r = np.array([-1.5,0])
s = np.array([1.5,0])
A = np.array([1/np.sqrt(2),1/np.sqrt(2)])
B = np.array([1,0])


x_points_1 = [np.cos(theta*np.pi/180) for theta in range(46)]
y_points_1 = [np.sin(theta*np.pi/180) for theta in range(46)] 

x_points_2 = [np.cos(theta*np.pi/180) for theta in range(46)]
y_points_2 = [0 for theta in range(46)]

x_points = x_points_1 + x_points_2
y_points = y_points_1 + y_points_2

x_lm = line_gen(l,m)
x_rs = line_gen(r,s)

len = 200
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = np.sqrt(1)*np.cos(theta)
x_circ[1,:] = np.sqrt(1)*np.sin(theta)
x_circ = (x_circ.T + O).T

plt.fill_between([0,1/np.sqrt(2),1],[0,1/np.sqrt(2),0],color = 'black')
plt.fill_between(x_points,y_points,color = 'black')

plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')
plt.plot(O[0],O[1],'o')
plt.plot(A[0],A[1],'o')
plt.plot(B[0],B[1],'o')
plt.plot(x_lm[0,:],x_lm[1,:])
plt.plot(x_rs[0,:],x_rs[1,:])

plt.text(O[0],O[1]+ 0.1,'O')
plt.text(A[0],A[1]+0.1,'A')
plt.text(B[0]+0.05,B[1]+0.05,'B')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')

plt.show()
