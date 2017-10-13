import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate


mod = SourceModule("""
	 #include <math.h>
	
    __global__ void PFSS_kern(float * theta_in, float * phi_in, float * r_in,
										float * g_in, float * h_in, float * outs, float * justB, float * aSCS_in)
		{
			//Using Schmidt normalized Legendre polynomials
			//Calculated using LOS coeffs (convert from rad before writing to GPU)
			//Assuming blockDim.x = ncoeffs + 1
			int my_m = threadIdx.x;
			float P = 1.0f;  //start at P(l=0,m=0)
			float dP = 0.0f; //dP(0,0)
			float c;  //coefficient variable 
			float theta = theta_in[blockIdx.x];
			float st = sin(theta);
			float ct = cos(theta);
			float phi = phi_in[blockIdx.x];
			float cmp = cos(my_m * phi);
			float smp = sin(my_m * phi);
			float r = r_in[blockIdx.x];  //beyond 2.5 Rs fall as R^-2 *****

			// advance to P(l=my_m, m=my_m)			
			if (my_m > 0) P = st;	// calc P(1,1) assuming my_m =/=0
			for (int mm1=1; mm1 < my_m; mm1++)  //sum over m minus 1 = 1 to my_m -1
			{
				c = sqrt((2.0f * mm1 + 1) / (2.0f * mm1 + 2)) * st;
				P *= c;
			}
			float Plm1 = P; //P(l-1,m) used for next P loop


			//calc and add first term (l=my_m, m=my_m) into B sums
			//need the 1D index of (l=my_m, m=my_m)
			int id1D = 0;
			for (int i=0; i<my_m; i++) id1D += blockDim.x - i;
			float g = g_in[id1D];
			float h = h_in[id1D];
				//B_radial calculation
			float Br_gh = g * cmp + h * smp;
			float Br_coeff = (my_m+1) * powf(r, -my_m-2); 		
			float Br_sum = Br_coeff * Br_gh * P;
				//B_phi calculation
			float Bp_coeff = powf(r, -my_m-2);
			float Bp_gh = g * smp - h * cmp;
			float Bp_sum = my_m * Bp_coeff * Bp_gh * P / st;
				//B theta calculation (need deriv of Legendre polys)
				//conveniently reuse parts of rad and theta calc
			dP = my_m * ct * P / st;
			float Bt_sum = -Bp_coeff * Br_gh * dP;
			
			//calc P(l=my_m+1, m=my_m) and dP
			P *= sqrt(2.0f * my_m + 1.0f) * ct;
			float Pl = P; //P(l,m), used for next loop
			//calc and sum corresponding B terms
			g = g_in[id1D+1];
			h = h_in[id1D+1];
			Br_gh = g * cmp + h * smp;
			Bp_gh = g * smp - h * cmp;
			int mp1 = my_m + 1;
			Br_coeff = 	(mp1+1) * powf(r, -mp1-2); 		
			Bp_coeff = powf(r, -mp1-2);
			float Br = Br_coeff * Br_gh * P;
			float Bp = my_m * Bp_coeff * Bp_gh * P / st;
			dP = (mp1 * ct * P -  sqrt(1.0f*mp1*mp1 - my_m*my_m) * Plm1) / st;
			float Bt = -Bp_coeff * Br_gh * dP;
			if (my_m < blockDim.x-1)  //don't sum for last thread
				{
					Br_sum += Br;
					Bp_sum += Bp;
					Bt_sum += Bt;
				}

			//calculate and sum up the l=my_m to l=lmax terms
			//loop for my_m = 0 through all but last two
			int gh_ind = id1D + 2;
			for (int l=my_m+1; l<blockDim.x-1; l++) {			
				c = sqrt((l+1.0f)*(l+1)- my_m *my_m);
				P = ((2*l+1)*ct*Pl - sqrt(l*l*1.0f - my_m*my_m)*Plm1)/c;
				//l for Plm calc is actually l-1 for dP and B calc
				int l2 = l + 1;
				dP = (l2 * ct * P -  sqrt(1.0f*l2*l2 - my_m*my_m) * Pl) / st;
				Plm1 = Pl;
				Pl = P;
				g = g_in[gh_ind];
				h = h_in[gh_ind];
				Br_gh = g * cmp + h * smp;
				Bp_gh = g * smp - h * cmp;
				Br_coeff = 	(l2+1) * powf(r, -l2-2); 		
				Bp_coeff = powf(r, -l2-2);
				Br = Br_coeff * Br_gh * P;
				Bp = my_m * Bp_coeff * Bp_gh * P / st;
				Bt = -Bp_coeff * Br_gh * dP;
				Br_sum += Br;
				Bp_sum += Bp;
				Bt_sum += Bt;
				gh_ind++;
			}
			__syncthreads();
	
			// have sum over l, copy into shared mem and have thread 0 sum over m
			// not the fastest method, but atomicAdd doesn't like floats
			// other option is to make a reduce -- won't help sh mem size but faster
			extern __shared__ float sh_B[];
			//cp to shared mem and adjust for r > 2.5 Rs (= mult by 1 if < 2.5)
			sh_B[my_m] = Br_sum; 
			__syncthreads();
			if (my_m ==1){
				int out_id = 4 * blockIdx.x;
				outs[out_id] = 0.0f;
			   for (int i=0; i<blockDim.x; i++) {
					outs[out_id] += sh_B[i];
					}
				}

			sh_B[my_m] = Bp_sum; 
			__syncthreads();
			if (my_m ==1){	
				int out_id = 4 * blockIdx.x + 2;
				outs[out_id] = 0.0f;
				for (int i=0; i<blockDim.x; i++) {
					outs[out_id] += sh_B[i];
					}
				}

			sh_B[my_m] = Bt_sum; 
			__syncthreads();
			if (my_m ==2){	
				int out_id = 4 * blockIdx.x + 1;
				outs[out_id] = 0.0f;
				for (int i=0; i<blockDim.x; i++) {
					outs[out_id] += sh_B[i];
					}	
			float b1 = outs[4*blockIdx.x];
			float b2 = outs[4*blockIdx.x+1];
			float b3 = outs[4*blockIdx.x+2];
			float bmag = sqrt(b1*b1 + b2*b2 + b3*b3);
			outs[4*blockIdx.x+3] = bmag;
			// artificial scaling * powf((1.0f / r_temp), -2)
			justB[blockIdx.x] =  bmag ; 
			}
			}
""")


global dtor, radeg
dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees


def initB(CR, npoints, ncoeffsLO, ncoeffsHI, R_SS):

	pycuda.tools.clear_context_caches()

	# Read in g and h coefficients as 2D arrays
	g2D = np.zeros([ncoeffsLO+1, ncoeffsLO+1])
	h2D = np.zeros([ncoeffsLO+1, ncoeffsLO+1])
	#fname = 'CR' + str(CR) + '_MDI' + str(ncoeffs) + '.dat'
	fname = '/home/cdkay/MagnetoPickles/SCScoeffs'+ str(CR) + '.pkl'
	f1 = open(fname, 'rb')
	GH = pickle.load(f1)
	f1.close()
	gLO2D = GH[0]
	hLO2D = GH[1]
	gHI2D = GH[2]
	hHI2D = GH[3]
		
	# Convert to 1D arrays
	global gLO, hLO, gHI, hHI 
	nelemsLO = np.sum(range(ncoeffsLO+2))
	gLO = np.zeros(nelemsLO)
	hLO = np.zeros(nelemsLO)
	nelemsHI = np.sum(range(ncoeffsHI+2))
	gHI = np.zeros(nelemsHI)
	hHI = np.zeros(nelemsHI)

	# No more normalizing here
	for l in range(ncoeffsLO+1):
		for m in range(ncoeffsLO+1):
			myID = m * (ncoeffsLO+1) - int(np.sum(range(m))) + l - m
			gLO[myID] = gLO2D[l,m]

	for l in range(ncoeffsLO+1):
		for m in range(ncoeffsLO+1):
			myID = m * (ncoeffsLO+1) - int(np.sum(range(m))) + l - m
			if m==0: hLO[myID] = 0
			else:
				hLO[myID] = hLO2D[l,m]
	for l in range(ncoeffsHI+1):
		for m in range(ncoeffsHI+1):
			myID = m * (ncoeffsHI+1) - int(np.sum(range(m))) + l - m
			gHI[myID] = gHI2D[l,m]

	for l in range(ncoeffsHI+1):
		for m in range(ncoeffsHI+1):
			myID = m * (ncoeffsHI+1) - int(np.sum(range(m))) + l - m
			if m==0: hHI[myID] = 0
			else:
				hHI[myID] = hHI2D[l,m]


	# Allocate g/h space on GPU and transfer LO (assume HI <= than in memory)
	gLO = gLO.astype(np.float32)
	hLO = hLO.astype(np.float32)
	gHI = gHI.astype(np.float32)
	hHI = hHI.astype(np.float32)
	global g_gpu, h_gpu
	g_gpu = cuda.mem_alloc(gLO.nbytes)
	cuda.memcpy_htod(g_gpu, gLO)
	h_gpu = cuda.mem_alloc(hLO.nbytes)
	cuda.memcpy_htod(h_gpu, hLO)

	# Set up memory for r, colat, lon
	global theta, theta_gpu, r, r_gpu, phi, phi_gpu
	theta = np.zeros(npoints)
	theta = theta.astype(np.float32)
	theta_gpu = cuda.mem_alloc(theta.nbytes)
	r = np.zeros(npoints)
	r = r.astype(np.float32)
	r_gpu = cuda.mem_alloc(r.nbytes)
	phi = np.zeros(npoints)
	phi = phi.astype(np.float32)
	phi_gpu = cuda.mem_alloc(phi.nbytes)
	
	# Set up memory for atomic sum and output
	a = np.zeros(ncoeffsLO+1)
	a = a.astype(np.float32)
	global ssize
	ssize = npoints * a.nbytes 

	# Set up outs to contain Br, Btheta, Bphi and Br for each point
	global outs, outs_gpu
	outs = np.zeros([4*npoints])
	outs = outs.astype(np.float32)
	outs_gpu = cuda.mem_alloc(outs.nbytes)
	cuda.memcpy_htod(outs_gpu, outs)

	# Set up justB to contain only B (less copying if not needed)
	global justB, justB_gpu
	justB = np.zeros([npoints])
	justB = justB.astype(np.float32)
	justB_gpu = cuda.mem_alloc(justB.nbytes)
	cuda.memcpy_htod(justB_gpu, justB)

	global RSS_gpu # ratio of rstar to rsun
	RSSar = np.array(R_SS, dtype=np.float32)
	RSS_gpu = cuda.mem_alloc(RSSar.size * RSSar.dtype.itemsize)
	cuda.memcpy_htod(RSS_gpu, RSSar)

	# Get GPU kernel
	global func
	func = mod.get_function("PFSS_kern")	

	# Calculate magnetic field at 2.5 Rsun
	#B105 = np.zeros([180,360])

	#Set up r and a
	global aSCS_gpu
	r = np.array([2.5]*npoints)
	r = r.astype(np.float32)
	r_gpu = cuda.mem_alloc(r.nbytes)
	cuda.memcpy_htod(r_gpu, r)
	aSCS = np.array([0.2]*npoints)
	aSCS = aSCS.astype(np.float32)
	aSCS_gpu = cuda.mem_alloc(aSCS.nbytes)
	cuda.memcpy_htod(aSCS_gpu, aSCS)

def calcB(R_in, Lat_in, Lon_in, dlon, npoints, LOorHI, ncoeffs, aSCS_in):
	if LOorHI == 0:
		cuda.memcpy_htod(g_gpu, gLO)
		cuda.memcpy_htod(h_gpu, hLO)
	if LOorHI == 1:
		cuda.memcpy_htod(g_gpu, gHI)
		cuda.memcpy_htod(h_gpu, hHI)

	#ncoeffs = 90
	#Transfer the thetas and phis to GPU
	r = np.array([R_in] * npoints, dtype=np.float32)
	theta = np.array([(90. - Lat_in) ] * npoints, dtype=np.float32) * dtor
	Lons = [Lon_in + dlon * i for i in range(npoints)]
	#print Lons
	phi = np.array(Lons, dtype=np.float32) * dtor
	cuda.memcpy_htod(theta_gpu, theta)
	cuda.memcpy_htod(phi_gpu, phi)
	cuda.memcpy_htod(r_gpu, r)
	cuda.memcpy_htod(aSCS_gpu, np.array([aSCS_in], dtype=np.float32))
	
	a = np.zeros(ncoeffs+1)
	a = a.astype(np.float32)
	ssize = a.nbytes * npoints

	func = mod.get_function("PFSS_kern")
	# Theta and phi should be in rads going into the kernel
	# R should be in solar radii
	func(theta_gpu, phi_gpu, r_gpu, g_gpu, h_gpu, outs_gpu, justB_gpu, aSCS_gpu, block=(ncoeffs+1, 1, 1), grid=(npoints,1,1), shared=ssize)
	cuda.memcpy_dtoh(outs, outs_gpu)
	result = outs#[]
	#for i in range(npoints):		
	#	temp = SPHVEC2CART(Lat_in, Lons[i], outs[0 + 4 * i], outs[1 + 4 * i], outs[2 + 4 * i])
	#	result.append(temp[0]) 
	#	result.append(temp[1])
	#	result.append(temp[2])
	#	result.append(outs[3 + 4 * i]) 
	return result



def SPH2CART(r, lat, lon):
	colatr = (90. - lat) * dtor
	lonr = lon * dtor
	x = r * np.sin(colatr) * np.cos(lonr)
	y = r * np.sin(colatr) * np.sin(lonr)
	z = r * np.cos(colatr)
	return x, y, z

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

def linterp(r_in, B_loR, B_hiR):
    iR = int((r_in- Rmin) / dR)
    r_grid = Rmin + dR * iR
    deltaB = (B_hiR - B_loR) * (r_in - r_grid) / dR
    newB = deltaB + B_loR
    # need to calc new magnitude
    newB[3] = np.sqrt(np.sum(newB[:3]**2))
    return newB

def slerp(iR, theta_in, phi_in, unifield):
    iTheta = int((180. - theta_in) / dAng)
    belowTheta = 180. - dAng * iTheta
    iPhi   = int(phi_in / dAng)
    dtT = (theta_in - belowTheta) / dAng
    frac_lo = np.sin(dtor * dAng * (1-dtT)) / np.sin(dtor*dAng)
    frac_hi = np.sin(dtor * dAng * dtT) / np.sin(dtor*dAng)
    iPhi2 = iPhi+1
    if iPhi2 >= nPhiHI -1: iPhi2 = 0  
    #print iR, iTheta, iPhi, iPhi2, theta_in, phi_in
    B_lophi = unifield[iR, iTheta+1, iPhi, :] * frac_hi + unifield[iR, iTheta, iPhi, :] * frac_lo
    B_hiphi = unifield[iR, iTheta+1, iPhi2, :] * frac_hi + unifield[iR, iTheta, iPhi2, :] * frac_lo
    # need to calc new magnitude
    B_lophi[3] = np.sqrt(np.sum(B_lophi[:3]**2))
    B_hiphi[3] = np.sqrt(np.sum(B_hiphi[:3]**2))
    dtP = (phi_in - iPhi * dAng) / dAng
    frac_lo = np.sin(dtor * dAng * (1-dtP)) / np.sin(dtor*dAng)
    frac_hi = np.sin(dtor * dAng * dtP) / np.sin(dtor*dAng)
    newBphid = frac_hi * B_hiphi + frac_lo * B_lophi
    # need to calc new magnitude
    newBphid[3] = np.sqrt(np.sum(newBphid[:3]**2))

    return newBphid
    
def getpickleB(r_in, theta_in, phi_in, unifield):
    iR = int((r_in- Rmin) / dR)
    B_loR = slerp(iR, theta_in, phi_in, unifield)
    B_hiR = slerp(iR+1, theta_in, phi_in, unifield)
    return linterp(r_in, B_loR, B_hiR)


def invert_field(unifield, Br_cusp):
    # convert cusp field to SPH from XYZ
    LO_shape = Br_cusp.shape #shape is theta, phi
    nThetaLO, nPhiLO = LO_shape[0], LO_shape[1]
    lats = np.linspace(-90, 90., nThetaLO)  
    lons = np.linspace(0, 359.5, nPhiLO) 

    # need to reduce cusp resolution to that of HI data
    # this is not most efficient method if LO is int mult of HI
    f = interpolate.interp2d(lons, lats, Br_cusp, kind='cubic')
    HI_shape = unifield.shape
    global nThetaHI, nPhiHI
    nThetaHI, nPhiHI = HI_shape[1], HI_shape[2] # HI shape is R, theta, phi, xyz
    lats2 = np.linspace(-90, 90, nThetaHI)  
    lons2 = np.linspace(0, 359.5, nPhiHI) 
    smallBr = np.zeros([nThetaHI,nPhiHI])
    for i in range(nThetaHI):
        for j in range(nPhiHI):
            smallBr[i,j] = f(lons2[j], lats2[i])

    # get the original polarity of unflipped field
    pols = np.zeros(smallBr.shape)-1
    pols[np.where(smallBr > 0)] = 1
    # set up array to hold traced polarities
    traced_pols = np.zeros([HI_shape[0], HI_shape[1], HI_shape[2]])
    traced_pols[0,:,:] = np.array(pols)
 
    # look for where the polarity flips to determine where to trace
    # rather than brute force tracing everywhere
    init_map = np.zeros(pols.shape)
    init_map = np.array(pols)
    nIters = 3 # more inters is more smoothing and more points to trace
    for k in range(nIters):
        init_map[1:-1,:] = init_map[0:-2,:] + init_map[2:,:]
        init_map[:,1:-1] = init_map[:,0:-2] + init_map[:,2:]
    # clean up the poles, assume no HCS there for now
    maxval = np.max(init_map)
    for k in range(nIters+1):
        init_map[k,:] = maxval* np.sign(init_map[4,:])
        init_map[-k,:] = maxval * np.sign(init_map[175,:])
        init_map[:,k] = init_map[:,nIters+2]
        init_map[:,-k] = init_map[:,-nIters-1]
 

    HCSids = np.where(np.abs(init_map) < maxval)
    print len(HCSids[0][:]), ' points to trace, beginning now....'

    global dR, Rmin, dtor, dAng
    dR   = 0.25  # unhardcode these later!!!!
    Rmin = 2.5
    dtor = math.pi / 180.
    dAng = 180. / (nThetaHI-1) # assume dAng same in Theta and Phi
    ds = 0.1

    # clean out anywhere Br < 0 (right near the HCS will get this)
    uniR = unifield[:,:,:,0]
    uniR[np.where(uniR <0)] = 1e-8#-uniR[np.where(uniR<0)]
    unifield[:,:,:,0] = uniR
    unifield[:,:,:,3] = np.sqrt(unifield[:,:,:,0]**2 + unifield[:,:,:,1]**2 + unifield[:,:,:,2]**2)

    # loop over the lines
    for i in range(len(HCSids[0][:])):
    #for i in range(10):
	id1, id2 = HCSids[0][i], HCSids[1][i]
    	myR = 2.50
    	if id1==90: id1 -=0.1 # will blow up if perfectly at equator due to 1/sintheta
    	myTheta = 180. - dAng * id1  # colat
    	myPhi   = dAng * id2
    	iTheta = id1
    	iPhi = id2
    	mypol = pols[iTheta, iPhi]
    	counter = 0

        iR = 0
   	while (myR < 25) and (counter < 9999):
		# sometimes it will find the poles to check, which we don't need to do
		if id1 >= 180: counter = 9999
		else:
			thisB = getpickleB(myR, myTheta, myPhi, unifield)
			#print myR, myTheta, myPhi, thisB
			if np.isreal(thisB[3]) and (thisB[3]>0):  # if trace outside somewhere just stop
				normB = thisB /thisB[3]
				if ds * normB[2] / myR/ np.sin(myTheta) < 0.5:
					myPhi   += ds * normB[2] / myR/ np.sin(myTheta)
				else:
					counter = 9999 # force it to skip since sinTheta blows up
				#print counter, iR, myR, myTheta, iTheta, iPhi, myPhi, ds * normB, np.sin(myTheta)
				myTheta += ds * normB[1] / myR
				myR     += ds * normB[0]
				iTheta = int((180. - myTheta) / dAng)
				iPhi = int(myPhi / dAng)
		       		if iPhi >= nPhiHI-1: 
			    		iPhi = 0
			    		myPhi -=360.
				iR = int((myR- Rmin) / dR)
			else: counter = 99999
			#print counter, iR, myR, myTheta, iTheta, iPhi, myPhi, ds * normB, np.sin(myTheta)
        	counter += 1
		if myR < 2.5: # going wrong dir
	 	    counter = 99999.
		    print "you're going the wrong way!  skipping step ", i
		# check if got large neg iVal, probably from sin(myTheta)~0 blowing things up
		if iPhi*iTheta < 0:
		    counter = 99999
		    print 'skipping step ', i, ' (sinTheta too large)'
		if counter > 9998:
		    myR = 9999.
		    print 'skipping step', i
		else:
			if (traced_pols[iR, iTheta, iPhi]==0) and (iR > 0): traced_pols[iR, iTheta, iPhi] = mypol
	print i, id1, id2, iTheta, iPhi

 
    # copy the last column over
    traced_pols[:,:,-1] = traced_pols[:,:,-2]

    for i in range(90):
   	# look for single points
   	this_slice = traced_pols[i+1,:,:]
   	abs_slice = np.abs(this_slice)
   	traced_points = np.where(this_slice != 0)
   	for j in range(len(traced_points[0])):
      		iTheta, iPhi = traced_points[0][j], traced_points[1][j]
      		if iPhi == 359: iPhi = -1 # make it play nice with neg indexing
      		neighbor_sum = abs_slice[iTheta, iPhi+1] + abs_slice[iTheta, iPhi-1] + abs_slice[iTheta+1, iPhi] + abs_slice[iTheta-1, iPhi]
      		if neighbor_sum == 0: 
         		this_slice[iTheta, iPhi] = 0
         		abs_slice[iTheta, iPhi]  = 0 
   	# go by lon and fill in
   	for j in range(nPhiHI):
      		nonzeros = np.where(this_slice[:,j] !=0)
      		if len(nonzeros[0]) > 0:
         		first_point, last_point = np.min(nonzeros), np.max(nonzeros)
         		if j==0: top_pol = this_slice[first_point,0]
         		this_slice[:first_point, j] = top_pol
         		this_slice[last_point+1:,j] = -1*top_pol 
      		if len(nonzeros[0]) > 2:
         		for k in range(len(nonzeros[0])-2):
            			if this_slice[nonzeros[0][k],j] == this_slice[nonzeros[0][k+1], j]:
               				this_slice[nonzeros[0][k]:nonzeros[0][k+1], j] = this_slice[nonzeros[0][k],j]
   	still_zero = np.where(this_slice == 0)
   	for k in range(len(still_zero[0])):
      		this_slice[still_zero[0][k], still_zero[1][k]] = traced_pols[i,still_zero[0][k], still_zero[1][k]]

   	traced_pols[i+1,:,:] = this_slice

    invert_field = np.zeros(unifield.shape)
    for i in range(3):
	invert_field[:,:,:,i] = unifield[:,:,:,i] * traced_pols

    return invert_field


def calcHCSdist(CR):
# This function calculates the position of the HCS based on the minimum in the magnetic field at 2.5 Rs.
# For each longitude we scan in latitude and indicate the minimum position in the bounds pickle with 
# -10.  If there are steep sections of the HCS we may miss them with this method so run a check on the
# difference in lat between any two adjacent points.
	# load the pickles
	f1 = open('SCS'+CR+'LOb_SPH.pkl', 'rb') # 2.5 rs 
	print "loading LOb pickle ..."
	B_high = pickle.load(f1)
	f1.close()
	fig = plt.figure()
	plt.imshow(B_high[-1,:,:,3], origin='lower')
	#plt.show()
	#f1 = open('CR'+str(CR)+'_bounds.pkl', 'wb')
	bounds = np.zeros([180,360])
	#f1.close()
	for i in range(bounds.shape[1]):
		if i > 0: old_min_lat = min_lat # track the last lat position
		Btemp = np.abs(B_high[-1,:,2*i,0])
		#print Btemp
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
	f1 = open('SCS'+str(CR)+'_bounds_HCS.pkl', 'wb')
	pickle.dump(bounds, f1)
	f1.close()
	fig = plt.figure(figsize=(9.0,7.0), dpi=100)
	plt.imshow(bounds, origin='lower')			
	plt.savefig('SCS'+str(CR)+'_bounds_HCS.png')
	# load the bounds pickle
	f1 = open('SCS'+str(CR)+'_bounds_HCS.pkl', 'rb')
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
	cax = ax.imshow(dists[:,:,1], origin='lower', vmin=0, vmax=50. )	
	fig.colorbar(cax, shrink=0.5)		
	plt.tight_layout()
	plt.savefig('SCS'+str(CR)+'_HCSdist.png')
	f1 = open('SCS'+str(CR)+'_dists.pkl', 'wb')
	pickle.dump(dists, f1)
	f1.close()

