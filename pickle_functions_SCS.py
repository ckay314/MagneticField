from __future__ import division
from matplotlib.patches import Patch

import numpy as np
import math

from pylab import *
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import pickle

import ForeCAT_functions as FC


# Christina Kay, January 2014
# This module contains a series of programs useful for manipulation the solar magnetic
# field calculated by makedatapickle.py which uses PFSS.py 
# The PFSS program runs on the GPU but rather than calculating the magnetic field on the fly
# for our CME deflection program ForeCAT we want to interpolate from predetermined data so that
# ForeCAT does not require GPU capabilities. Many of the tools we need for ForeCAT are useful 
#for other solar work so the programs are designed to stand alone from ForeCAT

# The magnetic field data is saved as a "pickle" which is just a way of storing an array for
# future use in Python.  To facilitate calculations we use a fixed grid so that we do not need 
# to search for points, the position determines the index in the array.  The pickles contain
# 4D arrays indexed as [r, lat, lon, B] where B is an array [Bx, By, Bz].  Cartesian coords
# are easier for me to think about in terms of vectors but we have even spacing in spherical 
# coords, hence the mixed coord systems.  There are programs to switch back and forth between
# coord systems. 

# Our standard assumed grid spacing is 0.01 Rs in R and 0.5 degrees in lat/lon
# r runs [1 -> 2.5]---- 1-1.75 in low pickle (76 pts), 1.75-2.5 in high (76 pts)
# lat runs [-90 -> 90] (361 pts)
# lon runs [0 -> 359.5] (720 pts, 0==360)
# We have a high and low pickle because the data files get too large otherwise


# these are global variables used by the programs
global rsun, dtor, radeg, kmRs, RSS, dR
rsun  =  7e10		 # convert to cm
dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees
kmRs  = 1.0e5 / rsun # km (/s) divided by rsun (in cm)
RSS = 2.5 # source surface distance
dR = (RSS - 1.0) / 150.


def init_pickle(CR):
# This program loads the pickles and MUST be called before any other pickle programs
# Depends on a Carrington Rotation number (int) and my naming convention for the high, low,
# and source surface (2.5 Rs) pickles.  It will not work if it does find the files with
# the correct names.
	# define global pickle vars
	global B_low, B_high, B_SS, B_105
	# get pickle name
	#fname = 'CR' + str(CR) 
	fname = 'SCSlow_' + str(CR) 
	# load the pickles
	f1 = open(fname+'a.pkl', 'rb') # 2.5 rs 
	print "loading low pickle ..."
	B_low = pickle.load(f1)
	f1.close()
	f1 = open(fname+'b.pkl', 'rb') # 2.5 rs 
	print "loading high pickle ..."
	B_high = pickle.load(f1)
	f1.close()
	f1 = open(fname+'c.pkl', 'rb') # 2.5 rs 
	print "loading SS pickle ..."
	B_SS = pickle.load(f1)
	f1.close()
	B_105 = B_low[6,:,:,:]
	# make arrays not lists
	B_low = np.array(B_low)
	B_high = np.array(B_high)
	B_SS = np.array(B_SS)
	B_105 = np.array(B_105)

	
def calcHCSpos(CR):
# This function calculates the position of the HCS based on the minimum in the magnetic field at 2.5 Rs.
# For each longitude we scan in latitude and indicate the minimum position in the bounds pickle with 
# -10.  If there are steep sections of the HCS we may miss them with this method so run a check on the
# difference in lat between any two adjacent points.
	#f1 = open('CR'+str(CR)+'_bounds.pkl', 'wb')
	bounds = np.zeros([180,360])
	#f1.close()
	for i in range(bounds.shape[1]):
		if i > 0: old_min_lat = min_lat # track the last lat position
		Btemp = B_high[-1,:,2*i,3]  
		print Btemp
		min_lat = int(np.argmin(Btemp) /2)
		if min_lat==180: min_lat -=1
		bounds[min_lat,i] = -10
		# check the spacing in lat between points adjacent in lon and fill in any gaps
		if i > 0:
			latdif = min_lat - old_min_lat
			if np.abs(latdif) > 1: # if points are not adjacent need to add to bounds
				halfdist = int((latdif/2))  # number of points to add for each lon
				for j in range(np.abs(halfdist)):
					k = j + 1
					if latdif < 0:
						bounds[old_min_lat - k, i-1] = -10
						bounds[min_lat + k, i] = -10
					if latdif > 0:
						bounds[old_min_lat + k, i-1] = -10
						bounds[min_lat - k, i] = -10
	f1 = open('CR'+str(CR)+'_bounds_HCS.pkl', 'wb')
	pickle.dump(bounds, f1)
	f1.close()
	fig = plt.figure(figsize=(9.0,7.0), dpi=100)
	plt.imshow(bounds, origin='lower')			
	plt.savefig('CR'+str(CR)+'_bounds_HCS.png')	
				
def calcHCSpos2(CR):
# This function calculates the position of the HCS based on the minimum in the magnetic field at 2.5 Rs.
# For each longitude we scan in latitude and indicate the minimum position in the bounds pickle with 
# -10.  If there are steep sections of the HCS we may miss them with this method so run a check on the
# difference in lat between any two adjacent points.
	#f1 = open('CR'+str(CR)+'_bounds.pkl', 'wb')
	bounds = np.zeros([180,360])
	#f1.close()
	f3 = open('xyz.pkl', 'rb')
	xyz = pickle.load(f3)
	f3.close()
	Bsign = np.zeros([361, 720])
	for i in range(361):
		for j in range(720):
			Bsign[i,j] = np.sign(B_high[-1,i,j,0] * xyz[i,j,0] +  B_high[-1,i,j,1] * xyz[i,j,1] 
					+  B_high[-1,i,j,2] * xyz[i,j,2]) 
	for i in range(bounds.shape[1]):
	#for i in range(1):
		#if i > 0: old_min_lat = min_lat # track the last lat position
		Btemp = Bsign[:,2*i]  
		#print Btemp
		switches = (Btemp[:-1] - Btemp[1:])
		#print switches 
		onlyswitches = np.where(switches != 0)
		for min_lat in onlyswitches[0]:
			#print min_lat
		#min_lat = int(np.argmin(Btemp) /2)
		#if min_lat==180: min_lat -=1
			bounds[int(min_lat/2),i] = -10
	for i in range(bounds.shape[0]):
		Btemp = Bsign[2*i,:]  
		switches = (Btemp[:-1] - Btemp[1:])
		onlyswitches = np.where(switches != 0)
		for min_lat in onlyswitches[0]:
			bounds[i,int(min_lat/2)] = -10
		# check the spacing in lat between points adjacent in lon and fill in any gaps
		'''if i > 0:
			latdif = min_lat - old_min_lat
			if np.abs(latdif) > 1: # if points are not adjacent need to add to bounds
				halfdist = int((latdif/2))  # number of points to add for each lon
				for j in range(np.abs(halfdist)):
					k = j + 1
					if latdif < 0:
						bounds[old_min_lat - k, i-1] = -10
						bounds[min_lat + k, i] = -10
					if latdif > 0:
						bounds[old_min_lat + k, i-1] = -10
						bounds[min_lat - k, i] = -10 '''
	f1 = open('CR'+str(CR)+'_bounds_HCS.pkl', 'wb')
	pickle.dump(bounds, f1)
	f1.close()
	fig = plt.figure(figsize=(9.0,7.0), dpi=100)
	plt.imshow(bounds, origin='lower')			
	plt.savefig('CR'+str(CR)+'_bounds_HCS.png')	


def calcHCSdists(CR):
# This function calculates the distance from both the CH boundary and the SB position.  We leave an 
# empty spot in the distance pickle which we later fill with the PS distance.  We use 1 deg resolution
# to save the distances and can slerp in between grid points.  All the distances are defined at the grid 
# midpoints (same as boundaries).  We start with a list of points corresponding to the CH boundaries or
# the HCS location.  We then go through each point and calculate distances to other grid points.  For
# any point once we reach another point with a distance lower than the newly calculated we stop proceeding
# in that direction
	# load the bounds pickle
	f1 = open('CR'+str(CR)+'_bounds_HCS.pkl', 'rb')
	bounds = pickle.load(f1)
	f1.close()
	# set up empty dist array [fromHCS, fromCH, fromPS(calc later)]
	dists = np.ones([180, 360, 3]) * 9999 # use absurdly high number to indicate unchecked
	# initiate lists that will hold all the starting points for calc'ing distances	
	SBlats = []
	SBlons = []
	for i in range(bounds.shape[0]):
		for j in range(bounds.shape[1]):
			if bounds[i,j] == -10: 
				SBlats.append(dtor * (i - 90.))
				SBlons.append(dtor * j)
				dists[i,j,1] = 0.
				dists[i,j,0] = 0. 
	# make numpy arrays
	SBlats = np.array(SBlats); SBlons = np.array(SBlons)
	# loop through lat and lon and find min distance from CH and SB
	for i in range(180):
		print i
		for j in range(360):
			# source surface lat and lon (radians)
			mylat = dtor * (i - 90.) ; mylon = dtor * j	
			# distances between SS position and HCS/SB
			HCSdists =  np.abs(np.arccos(np.sin(SBlats) * np.sin(mylat) + np.cos(SBlats) * np.cos(mylat) * np.cos(SBlons - mylon)))
			HCSdists = np.nan_to_num(HCSdists)
			#HCSdists = np.abs(SBlats - mylat)			
			minHCSdist = np.min(HCSdists[np.where(HCSdists >0)]) * radeg
			dists[i,j,1] = minHCSdist
	# save figures and pickle
	fig = plt.figure(figsize=(9.0,7.0), dpi=100)
	ax = fig.add_subplot(1,1,1)
	cax = ax.imshow(dists[:,:,1], origin='lower', norm = mpl.colors.Normalize(vmin=0, vmax=50.) )	
	fig.colorbar(cax, shrink=0.5)		
	plt.tight_layout()
	plt.savefig('CR'+str(CR)+'_HCSdist.png')
	f1 = open('CR'+str(CR)+'_dists.pkl', 'wb')
	pickle.dump(dists, f1)
	f1.close()

		


def SPHVEC2CART(lat, lon, vr, vt, vp):
	colatr = (90. - lat) * dtor
	lonr = lon * dtor
	vx = np.sin(colatr) * np.cos(lonr) * vr + np.cos(colatr) * np.cos(lonr) * vt - np.sin(lonr) * vp
	vy = np.sin(colatr) * np.sin(lonr) * vr + np.cos(colatr) * np.sin(lonr) * vt + np.cos(lonr) * vp
	vz = np.cos(colatr) * vr - np.sin(colatr) * vt
	return vx, vy, vz


def CART2SPH(x, y, z):
	R = np.sqrt(x**2 + y**2 + z**2)
	lat = 90. - math.acos(z / R) * radeg
	lon = math.atan(y / x) * radeg
	if lon < 0:
		if x < 0:
			lon += 180.
		elif x > 0:
			lon += 360. 
	elif lon > 0.:
		if x < 0:
			lon += 180. 
	return R, lat, lon


def haversin(theta):
# Calculate the haversin of an angle.  Assume input is in radians.
	hvs = np.sin(theta/2)**2
	return hvs


