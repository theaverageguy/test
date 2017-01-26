from numpy import dot
import numpy as np
def kf_predict(X, P, A, Q, B, U):
	X = dot(A, X) + dot(B, U)
	P = dot(A, dot(P, A.T)) + Q
	return(X,P)
from numpy import dot, sum, tile, linalg
from numpy.linalg import inv
def kf_update(X, P, Y, H, R):
 	IM = dot(H, X)
 	IS = R + dot(H, dot(P, H.T))
 	K = dot(P, dot(H.T, inv(IS)))
 	X = X + dot(K, (Y-IM))
 	P = P - dot(K, dot(IS, K.T))
 	LH = gauss_pdf(Y, IM, IS)
 	return (X,P,K,IM,IS,LH)
def gauss_pdf(X,M,S):
 	if M.shape()[1]==1:
 		DX = X - tile(M, X.shape()[1])
 		E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
 		E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
 		P = exp(-E)
 	elif X.shape()[1] == 1:
		DX = tile(X, M.shape()[1])- M
		E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
		E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
		P = exp(-E)
	else:
		DX = X-M
		E = 0.5 * dot(DX.T, dot(inv(S), DX))
		E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
		P = exp(-E)
	return (P[0],E[0])

datafile = open('data.txt')
data = datafile.read()
data = data.split('\n')
print len(data)
GPScord = []
for d in data:
	temp = d.split(',')
	temp2 = []
	temp2.append(float(temp[1][2:][:-1]))
	temp2.append(float(temp[2][2:][:-1]))
	temp2.append(float(temp[3][2:][:-2]))
	GPScord.append(temp2)
#for record in GPScord:
#	print record
from pykalman import KalmanFilter
from pykalman import UnscentedKalmanFilter
#kf = KalmanFilter(n_dim_obs=3)
kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
kf2 = UnscentedKalmanFilter()
measurements = GPScord
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lat = []
lon = []
alt = []
for sample in GPScord:
	lat.append(sample[0])
	lon.append(sample[1])
	alt.append(sample[2])
ax.scatter(lat, lon, alt, c='r', marker='o', s=10)
ax.plot(lat, lon, alt)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#plt.show()

#plt.figure(2)
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
One_Hz = []
One_Hz_Two = []
count = 0
while count<len(GPScord):
	if count%25 == 0 :
		One_Hz.append(GPScord[count])
		temp = []
		temp.append(GPScord[count][0])
		temp.append(GPScord[count][1])
		One_Hz_Two.append(temp)
	count = count + 1
print len(One_Hz)


lat = []
lon = []
alt = []
for sample in One_Hz:
	lat.append(sample[0])
	lon.append(sample[1])
	alt.append(sample[2])
ax.scatter(lat, lon, alt, c='r', marker='o', s=10)
ax.plot(lat, lon, alt)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#plt.figure(3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Five_Hz = []
count = 0
while count<len(GPScord):
	if count%5 == 0 :
		Five_Hz.append(GPScord[count])
	count = count + 1
print len(Five_Hz)

for sample in Five_Hz:
	lat.append(sample[0])
	lon.append(sample[1])
	alt.append(sample[2])
ax.scatter(lat, lon, alt, c='r', marker='o', s=10)
ax.plot(lat, lon, alt)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#plt.show()


fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
Ten_Hz = []
count = 0
while count<len(GPScord):
	if count%10 == 0 :
		Ten_Hz.append(GPScord[count])
	count = count + 1
print len(Ten_Hz)

lat = []
lon = []
alt = []
for sample in Ten_Hz:
	lat.append(sample[0])
	lon.append(sample[1])
	alt.append(sample[2])
ax.scatter(lat, lon, alt, c='r', marker='o', s=10)
ax.plot(lat, lon, alt)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#kf.em(One_Hz).smooth()
#kf2.smooth([91,3])
plt.show()
measurements = np.asarray(One_Hz_Two)
print kf.em(measurements).smooth([[2,0], [2,1], [2,2]])[0]
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
#print filtered_state_covariances
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
#means, covariances = kf.filter(np.asarray(One_Hz_Two))
#print smoothed_state_covariances



 	 