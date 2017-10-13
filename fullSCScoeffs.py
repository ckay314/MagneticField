import pyfits as pf
import numpy as np
import math
import pickle
import os.path
import sys
import matplotlib.pyplot as plt

def get_Legendre_polys(nTheta, nPhi, nHarmonics, mode='lat'):

	dPhi =2. * math.pi /nPhi
	dTheta = math.pi /nTheta
	dSinTheta = 2.0/nTheta

	# define arrays
	p_nm         = np.zeros([nHarmonics+1, nHarmonics+1])
	dp_nm        = np.zeros([nHarmonics+1, nHarmonics+1])
	pThetas      = np.empty([nHarmonics+1, nHarmonics+1, nTheta])
	dpThetas     = np.empty([nHarmonics+1, nHarmonics+1, nTheta])
	factorRatio  = np.zeros(nHarmonics+1)

	# calculate associated Legendre polynomials
	print "Calculating Legendre Polynomials..."
	print "Isn't this fun "
	print ''

	# array with factorial coefficient needed later
	factorRatio[0] = 1.0
	for ii in range(nHarmonics):
	    m = ii+1
	    factorRatio[m] = factorRatio[m-1] * np.sqrt(2*m-1) / np.sqrt(2*m)

	# Assuming sin latitude grid

	# Calculate the harmonic coefficients for each Theta value
	for iTheta in range(nTheta):
	    # THETA IS COLAT HERE
	    Theta = math.pi - (float(iTheta)+0.5) * dTheta
	    CosTheta = np.cos(Theta)
	    # calculate sin Theta, make sure not undefined
	    if np.sin(Theta) > 1e-9: SinTheta = np.sin(Theta)
	    else: SinTheta = 1e-9

	    if mode=='SinLat':
		CosTheta = -1. + (iTheta +0.5) * dSinTheta
		Theta = np.arccos(CosTheta)
		SinTheta = np.sin(Theta)
		#print iTheta, SinTheta, SinTheta2, Theta, Theta2, CosTheta, CosTheta2
			  
	    
	    # reset var that store SinTheta^m    
	    SinThetaM  = 1.0
	    
	    # empty out the p_nm array
	    p_nm[:,:] = 0.0
	    dp_nm[:,:] = 0.0
	    
	    for m in range(nHarmonics+1):
		if m==0: delta_m0 = 1
		else: delta_m0 = 0
		
		# fill in the diagonal terms    
		p_nm[m,m] = factorRatio[m] * np.sqrt((2-delta_m0)*(2*m+1)) * SinThetaM
		dp_nm[m,m] = m * CosTheta * p_nm[m,m] / SinTheta
		
		# fill in term one below the diagonal using recursive relation
		if m < nHarmonics: 
		    p_nm[m+1,m] = p_nm[m,m] * np.sqrt(2*m+3) * CosTheta
		    # div by sqrt(2l+1) at end, need to accound for different l in Plm terms used here
		    dp_nm[m+1,m] = ((m+1) * CosTheta * p_nm[m+1,m] - np.sqrt((m+1)**2 -m**2) * np.sqrt(2*(m+1) + 1) * p_nm[m,m]  / np.sqrt(2*m +1))/ SinTheta
		SinThetaM  = SinThetaM * SinTheta
		
	    # fill in the rest of the harmonic coeffs using recursive formula
	    for m in range(nHarmonics-1):
		for n in xrange(m+2, nHarmonics+1):
		    part1 = np.sqrt(2*n+1) / np.sqrt(n**2 - m**2)
		    part2 = np.sqrt(2*n-1)
		    part3 = np.sqrt((n-1)**2 - m**2) / np.sqrt(2*n-3)
		    p_nm[n,m]  = part1 * (part2 * CosTheta * p_nm[n-1,m] - part3 * p_nm[n-2,m])
		    # div by sqrt(2l+1) at end, need to accound for different l in Plm terms used here
		    part4 = np.sqrt(2*n+1) / np.sqrt(2*n-1)
		    dp_nm[n,m] = (n * CosTheta * p_nm[n,m] - np.sqrt(n**2 -m**2) * part4* p_nm[n-1,m])/ SinTheta
		  
	    # Apply Schmidt normalization
	    for m in range(nHarmonics+1):
		for n in xrange(m, nHarmonics+1):
		    part1 = 1. / np.sqrt(2*n + 1)
		    p_nm[n,m] = p_nm[n,m] * part1
		    pThetas[n,m,iTheta] = p_nm[n,m]    
		    dp_nm[n,m] = dp_nm[n,m] * part1
		    dpThetas[n,m,iTheta] = dp_nm[n,m]   
            
	return pThetas, dpThetas
                 

def calc_coeffs(nTheta, nPhi, nHarmonics, data):
    dPhi =2. * math.pi /nPhi
    dTheta = math.pi /nTheta
    dSinTheta = 2.0/nTheta
    pThetas, dpThetas = get_Legendre_polys(nTheta, nPhi, nHarmonics, mode='SinLat')
    g_LO         = np.zeros([nHarmonics+1, nHarmonics+1])
    h_LO         = np.zeros([nHarmonics+1, nHarmonics+1])
    CosMPhi      = np.empty([nPhi, nHarmonics+1])
    SinMPhi      = np.empty([nPhi, nHarmonics+1])
    tempPhiarr   = np.empty(nPhi)
    for iPhi in range(nPhi):
        for m in range(nHarmonics):
            CosMPhi[iPhi,m] = np.cos(m * (iPhi) * dPhi)
            SinMPhi[iPhi,m] = np.sin(m * (iPhi) * dPhi)

    # sum over the magnetogram to determine the harmonic coeffs
    da = dSinTheta * dPhi
    for n in range(nHarmonics+1):
        NormalizationFactor = (2. * n + 1.0) / (n + 1.0)
        for m in range(n+1):
           for iTheta in range(nTheta):
               # B*cos(mphi)
               tempPhiarr[:] = data[iTheta,:]*CosMPhi[:,m] 
               # for a pair n,m add in sum(Bcosmphi)*P(theta) for each theta
               g_LO[n,m] = g_LO[n,m] + np.sum(tempPhiarr) * da * pThetas[n,m,iTheta] 
               # B* sin(mphi)
               tempPhiarr[:] = data[iTheta,:]*SinMPhi[:,m] 
               # for a pair n,m add in sum(Bsinmphi)*P(theta) for each theta
               h_LO[n,m] = h_LO[n,m] + np.sum(tempPhiarr) * da * pThetas[n,m,iTheta] 
           # normalize                  
           g_LO[n,m] = NormalizationFactor * g_LO[n,m] / (4 * math.pi)
           h_LO[n,m] = NormalizationFactor * h_LO[n,m] / (4 * math.pi)
           print n, m, g_LO[n,m], h_LO[n,m]
           
    return g_LO, h_LO
    
def calc_coeffsLA(nTheta, nPhi, nHarmonics, data):
    dPhi =2. * math.pi /nPhi
    dTheta = math.pi /nTheta
    dSinTheta = 2.0/nTheta
    # B cusp array, stack by theta, then phi, the r/t/p
    Bcp = np.zeros([nTheta*nPhi*3,1])
    idx = 0
    # there are quicker ways than for loops!
    for rtp in range(3):
        for iTheta in range(nTheta):
            for iPhi in range(nPhi):
                Bcp[idx] = data[iTheta,iPhi,rtp]
                idx +=1

    # get Legendre polys (again?, depends on resolution)
    pThetas, dpThetas = get_Legendre_polys(nTheta, nPhi, nHarmonics)

    # the matrix from hell (alphabeta)
    # across phi, theta, rtp
    # down m,n, gh
    alphabeta = np.zeros([(nHarmonics+1)**2, nTheta*nPhi*3])
    #for n in range(nHarmonics+1):
    idx = 0
    phis = np.array([dPhi*(ii) for ii in range(nPhi)])
    row = np.zeros(nPhi*nTheta*3)
    # add alpha terms
    for n in range(nHarmonics+1):
        print n
        for m in range(n+1):
            alpha1 = []
            alpha2 = []
            alpha3 = []
            for iTheta in range(nTheta):
                #Theta = math.pi * 0.5 - np.arcsin((float(iTheta)+0.5) * dSinTheta - 1.0)
                Theta = math.pi - (float(iTheta)+0.5) * dTheta
                if np.sin(Theta) > 1e-9:
                    SinTheta = np.sin(Theta)
                else:
                    SinTheta = 1e-9            
                subalpha1 = (n+1) * pThetas[n,m,iTheta] * np.cos(m * phis)
                subalpha2 = - dpThetas[n,m,iTheta] * np.cos(m * phis)
                subalpha3 = (m / SinTheta) * pThetas[n,m,iTheta] * np.sin(m* phis)
                alpha1.append(subalpha1)
                alpha2.append(subalpha2)
                alpha3.append(subalpha3)  
            alpha1 = np.reshape(alpha1, -1)
            alpha2 = np.reshape(alpha2, -1)
            alpha3 = np.reshape(alpha3, -1)
            row = np.array([alpha1, alpha2, alpha3])
            alphabeta[idx,:] = np.reshape(row, -1)
            idx += 1
    # add beta terms
    for n in range(nHarmonics+1):
        print n
        for m in range(n+1):
            if m > 0:
                beta1 = []
                beta2 = []
                beta3 = []
                for iTheta in range(nTheta):
                    #Theta = math.pi * 0.5 - np.arcsin((float(iTheta)+0.5) * dSinTheta - 1.0)
                    Theta = math.pi - (float(iTheta)+0.5) * dTheta
                    if np.sin(Theta) > 1e-9:
                        SinTheta = np.sin(Theta)
                    else:
                        SinTheta = 1e-9            
                    subbeta1 = (n+1) * pThetas[n,m,iTheta] * np.sin(m * phis)
                    subbeta2 = - dpThetas[n,m,iTheta] * np.sin(m * phis)
                    subbeta3 = -(m / SinTheta) * pThetas[n,m,iTheta] * np.cos(m* phis)
                    beta1.append(subbeta1)
                    beta2.append(subbeta2)
                    beta3.append(subbeta3)            
                beta1 = np.reshape(beta1, -1)
                beta2 = np.reshape(beta2, -1)
                beta3 = np.reshape(beta3, -1)
                row = [beta1, beta2, beta3]
                alphabeta[idx,:] = np.reshape(row, -1)
                idx += 1

    # do the linear algebra
    # calculate AB = alphabeta dot alphabeta^T
    print 'calc AB'
    AB = np.dot(alphabeta, np.transpose(alphabeta))

    # calculate inverse of AB
    print 'calc inverse AB'
    AB_inv = np.linalg.inv(AB)

    # calculate alphabeta dot B
    print 'alphabeta dot B'
    abdotB = np.dot(alphabeta, Bcp)
    print np.sum(abdotB)
    # calculate the coeffs (inverse AB dot (ab dot B))
    print 'ABinv dot abdotB'
    GH = np.dot(AB_inv, abdotB)

    # unpack and print the coeffs
    idx = 0      
    g_HI = np.zeros([nHarmonics+1, nHarmonics+1])
    h_HI = np.zeros([nHarmonics+1, nHarmonics+1])      
    for n in range(nHarmonics+1):
        for m in range(n+1):
            print 'g', n, m, GH[idx]
            g_HI[n,m] = GH[idx]
            idx += 1
    for n in range(nHarmonics+1):
        for m in range(n+1):
            if m > 0:
                print 'h', n, m, GH[idx]
                h_HI[n,m] = GH[idx]
                idx += 1 
    
    return g_HI, h_HI    

# calculate the magnetic field at the cusp surface
def calc_B_surface(Rc, nTheta, nPhi, nHarmonics, g_in, h_in):
    dPhi =2. * math.pi /nPhi
    dTheta = math.pi /nTheta
    dSinTheta = 2.0/nTheta
    # redefine arrays
    CosMPhi      = np.empty([nPhi, nHarmonics+1])
    SinMPhi      = np.empty([nPhi, nHarmonics+1])
    tempPhiarr   = np.empty(nPhi)
    pThetas, dpThetas = get_Legendre_polys(nTheta, nPhi, nHarmonics)

    B_cusp = np.zeros([nTheta,nPhi,3])
    gcosphi = np.zeros([nHarmonics+1, nHarmonics+1, nPhi])
    gsinphi = np.zeros([nHarmonics+1, nHarmonics+1, nPhi])
    hcosphi = np.zeros([nHarmonics+1, nHarmonics+1, nPhi])
    hsinphi = np.zeros([nHarmonics+1, nHarmonics+1, nPhi])

    for iPhi in range(nPhi):
        for m in range(nHarmonics+1):
            CosMPhi[iPhi,m] = np.cos(m * (iPhi) * dPhi)
            SinMPhi[iPhi,m] = np.sin(m * (iPhi) * dPhi)
            gcosphi[:,m,iPhi] = g_in[:,m] * CosMPhi[iPhi,m]
            gsinphi[:,m,iPhi] = g_in[:,m] * SinMPhi[iPhi,m]
            hcosphi[:,m,iPhi] = h_in[:,m] * CosMPhi[iPhi,m]
            hsinphi[:,m,iPhi] = h_in[:,m] * SinMPhi[iPhi,m]
    PghR = np.zeros([nHarmonics+1, nHarmonics+1, nTheta, nPhi])
    PghT = np.zeros([nHarmonics+1, nHarmonics+1, nTheta, nPhi])
    PghP = np.zeros([nHarmonics+1, nHarmonics+1, nTheta, nPhi])
    for iTheta in range(nTheta):
        Theta = math.pi - (float(iTheta)+0.5) * dTheta
        if np.sin(Theta) > 1e-9: SinTheta = np.sin(Theta)
        else: SinTheta = 1e-9
        for n in range(nHarmonics+1):
            for m in range(n+1):
                PghR[n,m,iTheta,:] = pThetas[n,m,iTheta] * (gcosphi[n,m,:] + hsinphi[n,m,:])
                PghT[n,m,iTheta,:] = -dpThetas[n,m,iTheta] * (gcosphi[n,m,:] + hsinphi[n,m,:])
                PghP[n,m,iTheta,:] = m * pThetas[n,m,iTheta] * (gsinphi[n,m,:] - hcosphi[n,m,:]) / SinTheta

    sumPghR = PghR.sum(1)
    sumPghT = PghT.sum(1)
    sumPghP = PghP.sum(1)
    Br_unsum = np.zeros(sumPghR.shape)
    Bt_unsum = np.zeros(sumPghR.shape)
    Bp_unsum = np.zeros(sumPghR.shape)

    for n in range(nHarmonics + 1):
        Br_unsum[n,:,:] = (n+1) * (1/Rc)**(n+2) * sumPghR[n,:,:] 
        Bt_unsum[n,:,:] = (1/Rc)**(n+2) * sumPghT[n,:,:] 
        Bp_unsum[n,:,:] = (1/Rc)**(n+2) * sumPghP[n,:,:] 
    B_cusp[:,:,0] = Br_unsum.sum(0)
    B_cusp[:,:,1] = Bt_unsum.sum(0)
    B_cusp[:,:,2] = Bp_unsum.sum(0)
    return B_cusp

#----------------------------------------------------------------------------------

CRstr      = '2147'

# define filename
fname = 'CR'+CRstr+'.fits'

# check if file exists and either open fits or 
if os.path.isfile(fname):
    print 'Reading in ', fname
    myfits = pf.open(fname)
    orig_data = myfits[0].data
else:
    sys.exit( 'No file '+fname)

#get size of data
ny = orig_data.shape[0]
nx = orig_data.shape[1]
print 'Magnetogram size = ', ny, '(y) x', nx, ' (x)'
print ''

# check for various bad data flags
# clean out NaN/infs
good_points = np.isfinite(orig_data)
# assign bad points to zero
data = np.zeros([ny,nx])
data[good_points] = orig_data[good_points]

# check ranges to see if have more bad flags (ie super high neg or pos)
if np.min(data) < -5000: sys.exit('min < -5000, need to clean up data')
if np.max(data) >  5000: sys.exit('max > 5000, need to clean up data')

# set up constants for harmonic integration
nHarmonics = 90
nPhi = nx
nTheta = ny
 
g_LO, h_LO = calc_coeffs(nTheta, nPhi, nHarmonics, data)

# Calculate magnetic field at the cusp surface        

# Recalculate pTheta, dpTheta with lower resolution
nPhi = 180
nTheta = 90

pThetas, dpThetas = get_Legendre_polys(nTheta, nPhi, nHarmonics)


# calculate Br at the cusp
B_cusp = calc_B_surface(2.5, nTheta, nPhi, nHarmonics, g_LO, h_LO)

B_cusp = calc_B_surface(1, nTheta, nPhi, nHarmonics, g_LO, h_LO)
#fig = plt.figure()
#plt.imshow(B_cusp[:,:,0])
#plt.show()
B_cusp = calc_B_surface(2.5, nTheta, nPhi, nHarmonics, g_LO, h_LO)
#fig = plt.figure()
#plt.imshow(B_cusp[:,:,0])
#plt.show()

# Determine high coeffs----------------------------------------------------------------
# invert the magnetic field at the cusp
# array that stores initial polarity of Br at cus
negR = np.zeros(B_cusp.shape)
negR = negR[:,:,0]
negR[np.where(B_cusp[:,:,0] < 0)] = 1

# array with polarity flipped for positve Br
B_cusp_pos = B_cusp[:,:,:] # trick to make new copy
for i in range(3):
    Btemp = B_cusp[:,:,i]
    Btemp[np.where(negR == 1)] *= -1.
    B_cusp_pos[:,:,i] = Btemp

# calculate the hi coefficients by matching at boundary 
nHarmonics = 40
nPhi = 180
nTheta = 90

g_HI, h_HI = calc_coeffsLA(nTheta, nPhi, nHarmonics, B_cusp_pos)


# save a single pickle with all coeffs


all_coeffs = [g_LO, h_LO, g_HI, h_HI]
pickle.dump(all_coeffs, open('SCScoeffs'+CRstr+'.pkl', 'wb'))

# can check cusp surface from hi coeffs
check = 0
if check == 1:
	B_cusp = calc_B_surface(1, nTheta, nPhi, nHarmonics, g_HI, h_HI)
	fig = plt.figure()
	plt.imshow(B_cusp[:,:,0])
	plt.show()

