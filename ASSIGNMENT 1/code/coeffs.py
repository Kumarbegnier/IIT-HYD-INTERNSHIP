import numpy as np


def dir_vec(X,Y):
  return Y-X

def norm_vec(X,Y):
  return np.matmul(omat, dir_vec(X,Y))

#Generate line points
def line_gen(X,Y):
  len =10
  dim = X.shape[0]
  x_XY = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = X + lam_1[i]*(Y-X)
    x_XY[:,i]= temp1.T
  return x_XY

def tri_vert(x,y,z):
  p = (x**2 + z**2-y**2 )/(2*x)
  q = np.sqrt(z**2-p**2)
#Triangle vertices
  X = np.array([p,q]) 
  Y = np.array([0,0]) 
  Z = np.array([a,0]) 
  return  A,B,C


def line_dir_pt(m,X, dim):
  len = 10
  dim = X.shape[0]
  x_XY = np.zeros((dim,len))
  lam_1 = np.linspace(0,10,len)
  for i in range(len):
    temp1 = X + lam_1[i]*m
    x_XY[:,i]= temp1.T
  return x_XY

#Foot of the Altitude
def alt_foot(X,Y,Z):
  m = Y-Z
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@X 
  p[1] = n@Y
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P

#Intersection of two lines
def line_intersect(n1,X1,n2,X2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@X1
  p[1] = n2@X2
  #Intersection
  P=np.linalg.inv(N)@p
#  P=np.linalg.inv(N.T)@p
  return P

#Radius and centre of the circumcircle
#of triangle ABC
def ccircle(X,Y,Z):
  p = np.zeros(2)
  n1 = dir_vec(Y,X)
  p[0] = 0.5*(np.linalg.norm(A)**2-np.linalg.norm(Y)**2)
  n2 = dir_vec(Z,Y)
  p[1] = 0.5*(np.linalg.norm(B)**2-np.linalg.norm(Z)**2)
  #Intersection
  N=np.vstack((n1,n2))
  O=np.linalg.inv(N)@p
  r = np.linalg.norm(A -O)
  return O,r

#Radius and centre of the incircle
#of triangle ABC
def icentre(X,Y,Z,k1,k2):
  p = np.zeros(2)
  t = norm_vec(Y,Z)
  n1 = t/np.linalg.norm(t)
  t = norm_vec(Z,X)
  n2 = t/np.linalg.norm(t)
  t = norm_vec(X,Y)
  n3 = t/np.linalg.norm(t)
  p[0] = n1@Y- k1*n2@Z
  p[1] = n2@Z- k2*n3@X
  N=np.vstack((n1-k1*n2,n2-k2*n3))
  I=np.matmul(np.linalg.inv(N),p)
  r = n1@(I-Y)
  #Intersection
  return I,r

dvec = np.array([-1,1]) 
#Orthogonal matrix
omat = np.array([[0,1],[-1,0]]) 

