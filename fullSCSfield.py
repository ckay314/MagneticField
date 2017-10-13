from __future__ import division
from matplotlib.patches import Patch
import numpy as np
import SCS as Bmodel
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import pickle
import sys
import os.path
import pickle_functions_SCS as pf

# check number of inputs and request more if needed


# assign inputs to vars
CR = str(sys.argv[1])
sCR = str(CR)
Rcusp = float(sys.argv[2])
Rmax = float(sys.argv[3])
ncoeffsLO = int(sys.argv[4])
ncoeffsHI = int(sys.argv[5])
# not currently set up to include current (a!=0)
#acurrent  = float(sys.argv[6])

# constants
npoints = int(360/4) # number of concurrent points on GPU (div evenly into both nPHI!)
nThetaLO, nPhiLO = 361, 720
nThetaHI, nPhiHI = 181, 360
nRLO, nRHI = 76, 91 # nRLO for one of the half sections

# define edges of grid
latmin, latmax = -90., 90.
lonmin, lonmax = 0., 359.5

# set up arrays for grid
lats = np.linspace(latmin, latmax, nThetaLO)  
lons = np.linspace(lonmin, lonmax, nPhiLO) 
sublons = lons[::npoints]
dlon = lons[1] - lons[0] # spacing between lon points
rs   = np.linspace(1., Rcusp, 2*nRLO-1)

# Initialize background magnetic field
Bmodel.initB(CR, npoints, ncoeffsLO, ncoeffsHI, Rcusp)


# Pickle structure indexed [R, Lat, Lon, Bs]
# where Bs = [Bx, By, Bz, B]

# Low pickle, same pickle format as PFSS-------------------------------------------------------- 
# split into two portions <1.75 and 1.75<R<2.5Rs 
# for memory issues
dataa = np.empty([nRLO,nThetaLO,nPhiLO,4])
datab = np.empty([nRLO,nThetaLO,nPhiLO,4])
for i in range(nRLO*2-1):
#for i in range(1):
	#i = nRLO-1#nRLO*2-2
	R = rs[i]
	print sCR, R
	for j in range(nThetaLO):
		lat = lats[j]
		if lat == 90: lat=89.99
		for k in range(int(nPhiLO/npoints)):
			sublon = sublons[k]
			B = Bmodel.calcB(R, lat, sublon, dlon, npoints, 0, ncoeffsLO, 0.)
			for l in range(npoints):
				lonid = k * npoints + l
				if i <= nRLO-1:
					dataa[i,j,lonid,0] = B[0 + l * 4]
					dataa[i,j,lonid,1] = B[1 + l * 4]
					dataa[i,j,lonid,2] = B[2 + l * 4]
					dataa[i,j,lonid,3] = B[3 + l * 4]
					#print dataa[i,j, lonid], lat, lons[lonid]
				else:
					newi = i - nRLO +1
					if i ==2*nRLO: newi=nRLO-1
					datab[newi,j,lonid,0] = B[0 + l * 4]
					datab[newi,j,lonid,1] = B[1 + l * 4]
					datab[newi,j,lonid,2] = B[2 + l * 4]
					datab[newi,j,lonid,3] = B[3 + l * 4]
				# copy 75 in both so easy for interpolation
				if i == nRLO-1:
					datab[0,j,lonid,0] = B[0 + l * 4]
					datab[0,j,lonid,1] = B[1 + l * 4]
					datab[0,j,lonid,2] = B[2 + l * 4]
					datab[0,j,lonid,3] = B[3 + l * 4]

dataa = dataa.astype(np.float32)
datab = datab.astype(np.float32)

#fig = plt.figure()
#plt.imshow(dataa[0,:,:,0])
#plt.colorbar()
#plt.show()

# Unipolar high field, use lower resolution than low field -> only need single pickle-------
datac = np.empty([nRHI,nThetaHI,nPhiHI,4])
lats = np.linspace(latmin, latmax, nThetaHI)  
lons = np.linspace(lonmin, lonmax, nPhiHI) 
sublons = lons[::npoints]
dlon = lons[1] - lons[0] # spacing between lon points
rs   = np.linspace(Rcusp, Rmax, nRHI) / Rcusp
#anorm = acurrent / Rcusp # normalize by Rcusp

for i in range(nRHI):
#for i in range(1):
	R = rs[i]
	print sCR, R*Rcusp
	for j in range(nThetaHI):
		lat = lats[j]
		if lat == 90: lat=89.99
		for k in range(int(nPhiHI / npoints)): 
			sublon = sublons[k]
			B = Bmodel.calcB(R, lat, sublon, dlon, npoints, 1, ncoeffsHI, 0.)
			for l in range(npoints):
				lonid = k * npoints + l
				datac[i,j,lonid,0] = B[0 + l * 4]
				datac[i,j,lonid,1] = B[1 + l * 4]
				datac[i,j,lonid,2] = B[2 + l * 4]
				datac[i,j,lonid,3] = B[3 + l * 4]

# Invert high field-------------------------------------------------------------------------
dataa = dataa.astype(np.float32)
print 'saving ', 'SCS' + sCR +'LOa_SPH.pkl'
fa = open('SCS' + sCR +'LOa_SPH.pkl', "wb")
pickle.dump(dataa, fa)
fa.close()
print 'saving ', 'SCS' + sCR +'LOb_SPH.pkl'
datab = datab.astype(np.float32)
fb = open('SCS' + sCR +'LOb_SPH.pkl', 'wb')
pickle.dump(datab, fb)
fb.close()				
#print 'saving ', 'SCS' + sCR +'HIuni.pkl'
#fc = open('SCS_' + sCR +'HIuni.pkl', 'wb')
#pickle.dump(datac, fc)

datad = Bmodel.invert_field(datac, datab[-1,:,:,0])

datad = datad.astype(np.float32)
print 'saving ', 'SCS' + sCR +'HI_SPH.pkl'
fc = open('SCS' + sCR +'HI_SPH.pkl', 'wb')
pickle.dump(datad, fc)


#fig = plt.figure()
#plt.imshow(datab[-1,:,:,0])
#plt.colorbar()
#plt.show()

# convert SPH to cart
if os.path.isfile('SPH2CARTLO.pkl'):
	SPH2CARTmatLO = pickle.load(open('SPH2CARTLO.pkl', 'rb'))
else:
	SPH2CARTmatLO = np.zeros([nThetaLO, nPhiLO, 8])
	lats = np.linspace(latmin, latmax, nThetaLO) * 3.14159 / 180.
	lons = np.linspace(lonmin, lonmax, nPhiLO)  * 3.14159 / 180.
	for i in range(nThetaLO):
	   colatr = (3.14159/2 - lats[i])
	   for j in range(nPhiLO):
	      lonr = lons[j]
	      SPH2CARTmatLO[i,j,0:3] = [np.sin(colatr) * np.cos(lonr), np.cos(colatr) * np.cos(lonr), -np.sin(lonr)]
	      SPH2CARTmatLO[i,j,3:6] = [np.sin(colatr) * np.sin(lonr), np.cos(colatr) * np.sin(lonr),  np.cos(lonr)]
	      SPH2CARTmatLO[i,j,6:]  = [np.cos(colatr), -np.sin(colatr)]
	f1 = open('SPH2CARTLO.pkl', "wb")
	pickle.dump(SPH2CARTmatLO, f1)
	f1.close()

if os.path.isfile('SPH2CARTHI.pkl'):
	SPH2CARTmatHI = pickle.load(open('SPH2CARTHI.pkl', 'rb'))
else:
	SPH2CARTmatHI = np.zeros([nThetaHI, nPhiHI, 8])
	lats = np.linspace(latmin, latmax, nThetaHI) * 3.14159 / 180.
	lons = np.linspace(lonmin, lonmax, nPhiHI)  * 3.14159 / 180.
	for i in range(nThetaHI):
	   colatr = (3.14159/2 - lats[i])
	   for j in range(nPhiHI):
	      lonr = lons[j]
	      SPH2CARTmatHI[i,j,0:3] = [np.sin(colatr) * np.cos(lonr), np.cos(colatr) * np.cos(lonr), -np.sin(lonr)]
	      SPH2CARTmatHI[i,j,3:6] = [np.sin(colatr) * np.sin(lonr), np.cos(colatr) * np.sin(lonr),  np.cos(lonr)]
	      SPH2CARTmatHI[i,j,6:]  = [np.cos(colatr), -np.sin(colatr)]
	f1 = open('SPH2CARTHI.pkl', "wb")
	pickle.dump(SPH2CARTmatHI, f1)
	f1.close()


for i in range(nRLO):
   myBr = dataa[i,:,:,0]
   myBt = dataa[i,:,:,1]
   myBp = dataa[i,:,:,2]
   Bx = myBr * SPH2CARTmatLO[:,:,0] + myBt * SPH2CARTmatLO[:,:,1] + myBp * SPH2CARTmatLO[:,:,2]
   By = myBr * SPH2CARTmatLO[:,:,3] + myBt * SPH2CARTmatLO[:,:,4] + myBp * SPH2CARTmatLO[:,:,5]
   Bz = myBr * SPH2CARTmatLO[:,:,6] + myBt * SPH2CARTmatLO[:,:,7]
   dataa[i,:,:,0] = Bx 
   dataa[i,:,:,1] = By
   dataa[i,:,:,2] = Bz
   myBr = datab[i,:,:,0]
   myBt = datab[i,:,:,1]
   myBp = datab[i,:,:,2]
   Bx = myBr * SPH2CARTmatLO[:,:,0] + myBt * SPH2CARTmatLO[:,:,1] + myBp * SPH2CARTmatLO[:,:,2]
   By = myBr * SPH2CARTmatLO[:,:,3] + myBt * SPH2CARTmatLO[:,:,4] + myBp * SPH2CARTmatLO[:,:,5]
   Bz = myBr * SPH2CARTmatLO[:,:,6] + myBt * SPH2CARTmatLO[:,:,7]
   datab[i,:,:,0] = Bx
   datab[i,:,:,1] = By
   datab[i,:,:,2] = Bz

dataa[:,:,:,3] = np.sqrt(dataa[:,:,:,0]**2 + dataa[:,:,:,1]**2 + dataa[:,:,:,2]**2)
datab[:,:,:,3] = np.sqrt(datab[:,:,:,0]**2 + datab[:,:,:,1]**2 + datab[:,:,:,2]**2)



for i in range(nRHI):
   myBr = datad[i,:,:,0]
   myBt = datad[i,:,:,0]
   myBp = datad[i,:,:,0]
   Bx = myBr * SPH2CARTmatHI[:,:,0] + myBt * SPH2CARTmatHI[:,:,1] + myBp * SPH2CARTmatHI[:,:,2]
   By = myBr * SPH2CARTmatHI[:,:,3] + myBt * SPH2CARTmatHI[:,:,4] + myBp * SPH2CARTmatHI[:,:,5]
   Bz = myBr * SPH2CARTmatHI[:,:,6] + myBt * SPH2CARTmatHI[:,:,7]
   datad[i,:,:,0] = Bx
   datad[i,:,:,1] = By
   datad[i,:,:,2] = Bz

datad[:,:,:,3] = np.sqrt(datad[:,:,:,0]**2 + datad[:,:,:,1]**2 + datad[:,:,:,2]**2)

#fig = plt.figure()
#plt.imshow(datad[0,:,:,0])
#plt.colorbar()
#plt.show()



# Save pickles	
print 'saving ', 'SCS' + sCR +'LOa.pkl'
dataa = dataa.astype(np.float32)
fa = open('SCS' + sCR +'LOa.pkl', "wb")
pickle.dump(dataa, fa)
fa.close()
print 'saving ', 'SCS' + sCR +'LOb.pkl'
datab = datab.astype(np.float32)
fb = open('SCS' + sCR +'LOb.pkl', 'wb')
pickle.dump(datab, fb)
fb.close()				
print 'saving ', 'SCS' + sCR +'HI.pkl'
datac = datac.astype(np.float32)
fc = open('SCS' + sCR +'HI.pkl', 'wb')
pickle.dump(datad, fc)
fc.close()


Bmodel.calcHCSdist(CR)

