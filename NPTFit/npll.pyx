###############################################################################
# npll.pyx
###############################################################################
#
# Calculation of non-poissonian contribution to Log Likelihood, which is then
# combined with the poissonian contribution using the method of generating
# functions.
#
# Calculation broken into two parts:
#  1. Determination of x_m and x_m_sum, which depends on the number of 
#     non-poissonian templates & the number of breaks each one has; and
#  2. Determination of LL in terms of these using a recurrence relation
#
###############################################################################

import sys
import numpy as np
cimport numpy as np
cimport cython
import x_m
import scipy.integrate as integrate
from scipy.misc import factorial
from scipy.special import binom
from libc.stdlib cimport malloc, free

# Type used for all non-integer functions
DTYPE = np.float

# Setup cython functions
cdef extern from "math.h":
    double log(double x) nogil
    double exp(double x) nogil


def log_like(pt_sum_compressed, theta, f_ary, df_rho_div_f_ary, npt_compressed,
             data):
    """ Python wrapper for the log likelihood

    Organises the calculation of x_m values for multiple non-poissonian 
    templates

    Args:
        pt_sum_compressed: pixel-wise sum of Poissonian model templates
        theta: Array of non-poissonian parameters. Length of array is the
               number of NP templates, and each element in the array has
               the form: [A, n[1], .., n[j+1], Sb[1], .., Sb[j], Ebins, Edep, Eparams]
        f_ary: Photon leakage probabilities characterizing PSF, sum(f_ary) = 1.0
        df_rho_div_f_ary: df*rho(f)/f for integrating over f as a sum
        npt_compressed: Array of non-poissonian templates. Length of array is
                        the number of PS templates, and each element in the
                        array is the pixel-wise normalization of the PS template
        data: The pixel-wise data maps

    Returns:
        double log likelihood

    """
    cdef int nTheta = len(theta)
    cdef int nBins = len(theta[0,-3]) - 1

    cdef int[::1] data_sum = np.array(map(sum,zip(*data)),dtype=np.int32)
    cdef int[::1] k_max_bin = np.array(map(max,data+1),dtype=np.int32) 		
    cdef int k_max = np.max(data_sum) + 1
    cdef int k_max_comb = np.max(k_max_bin)

    cdef int npixROI = len(pt_sum_compressed)
    cdef double[:,:,::1] x_m_ary = np.zeros((nTheta,npixROI,k_max + 1), dtype=DTYPE)
    cdef double[:,::1] x_m_sum = np.zeros((nTheta,npixROI), dtype=DTYPE)

    cdef Py_ssize_t i=0,b=0
    cdef double[:,::1] x_m_ary_out = np.zeros((npixROI,k_max + 1), dtype=DTYPE)
    cdef double[::1] x_m_sum_out = np.zeros(npixROI, dtype=DTYPE)

    cdef double[::1] norm = np.zeros(nTheta, dtype=DTYPE)
    cdef double[:,::1] lambda_i = np.zeros((nTheta, nBins), dtype=DTYPE)

    for i in range(nTheta):
        # Check theta has the correct length
        assert( (len(theta[i])-3) % 2 == 0), "theta has an invalid length!"

        x_m_ary_out, x_m_sum_out = x_m.return_xs(theta[i,:-3].astype(DTYPE), f_ary, df_rho_div_f_ary, 
                                       npt_compressed[i], data_sum)

        x_m_ary[i] = x_m_ary_out
        x_m_sum[i] = x_m_sum_out

        norm[i] = integrate.quad( lambda x: theta[i,-2](x,theta[i,-1]), theta[i,-3][0], theta[i,-3][nBins] )[0]
        for b in range(nBins):
            lambda_i[i,b] = integrate.quad( lambda x: theta[i,-2](x,theta[i,-1]), theta[i,-3][b], theta[i,-3][b+1] )[0]/norm[i]

    return log_like_internal(pt_sum_compressed, data, data_sum, x_m_ary, x_m_sum, k_max,
                             npixROI, lambda_i)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double log_like_internal(double[::1] pt_sum_compressed, int[:,::1] data, int[::1] data_sum,
                              double[:,:,::1] x_m_ary, double[:,::1] x_m_sum, 
                              int k_max, int npixROI, double[:,::1] lambda_i):
    """ Calculation of log likelihood

    Take the x_m and x_m_sum values, which represent the information for the
    non-poissonian templates, combine these with sum of the poissonian
    templates (PT_sum) and calculate the full log likelihood.

    Calculation for the likelihood proceeds via a recursion relation for the pk
    values.

    Returns:
        double log likelihood 

    """
    cdef int nTemp = len(x_m_sum)
    cdef int nBins = len(lambda_i[0])

    cdef Py_ssize_t p=0, k=0, n=0, i=0, j=0, t=0, k1=0
    cdef double ll = 0.
    cdef double f0_ary=0, f1_ary=0
    cdef double[:,::1] pk = np.zeros((nTemp,k_max+1), dtype=DTYPE)

    cdef double term = 0.
    cdef double term_comb = 0.
 
    cdef int[::1] comb_off = np.zeros(k_max+1, dtype=np.int32)
    cdef int[::1] comb_len = np.zeros(k_max+1, dtype=np.int32)

    comb_len[0] = 1
    for n in range(k_max):
        comb_len[n+1] = <int>binom(nTemp+n,nTemp-1)
        comb_off[n+1] = comb_off[n] + comb_len[n] 

    cdef int[:,::1] comb_list = np.zeros((comb_off[k_max]+comb_len[k_max],nTemp),dtype=np.int32)
    cdef int offset = 0
    for n in range(k_max+1):
        RecAssign(nTemp,&offset,comb_list,0,n)

    cdef double[:,:,::1] powLambda = np.zeros((nTemp,nBins,k_max+1), dtype=DTYPE)
    for t in range(nTemp):
        for n in range(nBins):
            for k in range(k_max+1):
                powLambda[t][n][k] = lambda_i[t][n]**k 
 
    cdef int[::1] comb_beta = np.zeros( nBins, dtype=np.int32)
    cdef double result = 0

    # Loop over pixels
    for p in range(npixROI):
        for t in range(nTemp):
            # Define p_0 (pk[0]) and p_1 (pk[1])
            # Then the remaining p_k are determined recursively up to the value of 
            # k = data in that pixel
            f0_ary = -(pt_sum_compressed[p] + x_m_sum[t,p])
            f1_ary = (pt_sum_compressed[p] + x_m_ary[t,p,1])
            pk[t][0] = exp(f0_ary)
            pk[t][1] = pk[t][0] * f1_ary

            for k in range(2, k_max+1):
                pk[t][k] = 0.
                for n in range(0, k-1):
                    pk[t][k] += (k-n)/float(k)*x_m_ary[t,p,k-n]*pk[t][n]
                pk[t][k] += f1_ary*pk[t][k-1]/float(k)

            result = 0
            RecBeta(0,nBins,nTemp, p, comb_off, comb_len, comb_list, comb_beta, pk, powLambda, data, &result)

        if result > 0:
            ll += log(result)
        else:
            ll += -10.1**10.

    return ll

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef RecBeta(int b,int nBins,int nTemp,int p, int[::1] comb_off, int[::1] comb_len, int[:,::1] comb_list, int[::1] comb_beta, 
             double[:,::1] pk, double[:,:,::1] powLambda, int[:,::1] data, double *res):
    cdef double term = 1.
    cdef int[:,::1] beta = np.zeros((nTemp,nBins), dtype=np.int32)
    cdef int[::1] beta_sum = np.zeros(nTemp,dtype=np.int32)

    if nBins == b:
        for i in range(nTemp):
            for j in range(nBins):
                beta[i][j] = comb_list[ comb_off[ data[j,p] ] + comb_beta[j] ][i]
            beta_sum[i] = sum(beta[i])

            term *=  MultiNomCoeff(beta[i]) * pk[i][ beta_sum[i] ]
            for j in range(nBins):
                term *= powLambda[i][j][ beta[i][j] ]
        res[0] += term
    else:
        for i in range(comb_len[data[b,p]]):
            comb_beta[b] = i
            RecBeta( b+1, nBins, nTemp, p, comb_off, comb_len, comb_list, comb_beta, pk, powLambda, data, &res[0] )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef double MultiNomCoeff(int[:] comb):
    cdef Py_ssize_t a,j
    cdef double res=1, i = 1
    for a in range(len(comb)):
        for j in range(1,comb[a]+1):
            res *= i
            res //= j
            i += 1
    return res 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
cdef RecAssign(int nTemp,int *offset,int[:,::1] comb_list,int temp,int k):
    cdef int off_loc
    if nTemp-1 == temp:
        comb_list[offset[0]][temp] = k
        offset[0] += 1 
    else:
        for n in range(k+1):
            off_loc = offset[0]
            RecAssign(nTemp,&offset[0],comb_list,temp+1,k-n)
            for s in range(off_loc,offset[0]):
                comb_list[s][temp] = n 

