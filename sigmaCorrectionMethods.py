# core functions for the implementation of the proposed estimators

import numpy
import scipy
import time

import sklearn.covariance
        

# checked
def getAsymptoticCorrectedSigma(inlierAbsDiff, n):

    m = inlierAbsDiff.shape[0]
    inlierRatio = m / n

    if m == n:
        return numpy.sqrt(numpy.mean(numpy.square(inlierAbsDiff)))
    else:    
        correctionFactor = 1.0 / scipy.stats.chi2.ppf(inlierRatio, df = 1.0)

        empiricalQuantile = numpy.max(inlierAbsDiff)
        
        correctedSigma = empiricalQuantile * numpy.sqrt(correctionFactor)
        return correctedSigma


# m = number of inliers
# n = total number of samples
def estimateFiniteCorrectionFactors(m, n, nrSamplesForEstimation):
    assert(m < n)
    assert(m <= 10000) # otherwise might be too slow
    assert(nrSamplesForEstimation >= 100000)

    allSamples_collected = numpy.zeros((nrSamplesForEstimation, m))
    for i in range(nrSamplesForEstimation):
        allSamples = scipy.stats.norm.rvs(loc=0.0, scale=1.0, size = n)
        
        allSamplesSquared = numpy.square(allSamples)
        inlierSamplesSquared = numpy.sort(allSamplesSquared)[0:m]

        allSamples_collected[i] = inlierSamplesSquared

    allVariances, allExpectations, best_a, nu = getStatisticsForOptEstimator(allSamples_collected)

    allNormalizedVariances = allVariances / numpy.square(allExpectations)
    smallestVarId = numpy.argmin(allNormalizedVariances)
    correctionFac_smallestVarId = allExpectations[smallestVarId]

    assert(allExpectations.shape[0] == m)
    correctionFac_largestInlier = allExpectations[m-1]
    correctionFac_mean = numpy.mean(allExpectations)

    return correctionFac_mean, correctionFac_largestInlier, smallestVarId, correctionFac_smallestVarId, allExpectations, best_a, nu




def getQn_m(n):
    nHalfPlus = (n // 2) + 1
    m = (nHalfPlus * (nHalfPlus - 1) // 2)
    return m

# checked (returns same result up to 3 digits as robustbase.Qn(allSamples, finite_corr = False))
# my implmentation of Q_n according to "Alternatives to the Median Absolute Deviation", JASSA, 1993
def myQn(obs):
    
    n = obs.shape[0]

    m = getQn_m(n)
    all_squared_distances_sorted = getPairwiseSquaredDistanceSorted(obs)

    correctionFac = 1 / (numpy.sqrt(2) * scipy.stats.norm.ppf(q = 5 /8))
    
    variance_estimate = all_squared_distances_sorted[m - 1] * (correctionFac ** 2)
    return variance_estimate




def getStatisticsForOptEstimator(allSamples_collected, sigmaEstimate = None):
    
    allExpectations = numpy.mean(allSamples_collected, axis = 0)
    allVariances = numpy.var(allSamples_collected, ddof = 1, axis = 0)

    allSamples_normalized = allSamples_collected / allExpectations
    
    if sigmaEstimate is None:
        sigmaEstimate = sklearn.covariance.LedoitWolf(store_precision=False).fit(allSamples_normalized).covariance_ # running on mini2 (for n = 200, 300)
        # sigmaEstimate = numpy.cov(allSamples_normalized, rowvar = False, bias = False)
        # sigmaEstimate = sklearn.covariance.LedoitWolf(store_precision=False).fit(allSamples_normalized).covariance_ # running on mini2 (for n = 200, 300)
        # sigmaEstimate = sklearn.covariance.OAS(store_precision=False).fit(allSamples_normalized).covariance_ # running on mini1 (for n = 200, 300)
        assert(False)

    print("*** so far so good *** ")
    best_a = optimalLinearCombination(sigmaEstimate)
    nu = best_a.transpose() @ sigmaEstimate @ best_a

    return allVariances, allExpectations, best_a, nu





def getPairwiseSquaredDistanceSorted(allSamples):
    n = allSamples.shape[0]

    X = allSamples.reshape((n, 1))
    all_squared_distances = scipy.spatial.distance.pdist(X, 'sqeuclidean')
    all_squared_distances_sorted = numpy.sort(all_squared_distances)
    return all_squared_distances_sorted


# def estimateFiniteCorrectionFactors_Qn(nrAllObservations_n, nrSamplesForEstimation, useMemoryEfficient):
#     assert(nrAllObservations_n <= 500) # otherwise might be too slow
#     assert(nrSamplesForEstimation >= 100000)
    
#     m = getQn_m(nrAllObservations_n)

#     print("nrSamplesForEstimation = ", nrSamplesForEstimation)
#     print("m = ", m)
    
    
#     if not useMemoryEfficient:
#         assert(False)
#         allSamples_collected = numpy.zeros((nrSamplesForEstimation, m))
#         for i in range(nrSamplesForEstimation):
#             if i % 100 == 0:
#                 print("i = ", i)
#             allSamples = scipy.stats.norm.rvs(loc=0.0, scale=1.0, size = nrAllObservations_n)
            
#             all_squared_distances_sorted = getPairwiseSquaredDistanceSorted(allSamples)
            
#             # print("all_squared_distances_sorted = ", all_squared_distances_sorted.shape[0])
#             # print("scipy.misc.comb(N,k) = ", scipy.special.comb(n,2))
            
#             allSamples_collected[i] = all_squared_distances_sorted[0:m]
#             del allSamples
#             del all_squared_distances_sorted

#         print("** SO FAR SUCCESSFUL **")
#         gc.collect()
#         _, allExpectations, best_a, nu = getStatisticsForOptEstimator(allSamples_collected)
        
#     else:
#         # Implements the Rao-Blackwell-Ledoit-Wolf Estimator from "Shrinkage Algorithms for MMSE Covariance Estimation", 2009
        
#         print("USE HOME-MADE COV MATRIX ESTIMATION")
#         BATCH_SIZE = 1000
        
#         assert((nrSamplesForEstimation % BATCH_SIZE) == 0)

#         start_time = time.time()

#         allExpectations = numpy.zeros(m)

#         S_hat = numpy.zeros((m,m))
#         allSamples_collected = numpy.zeros((BATCH_SIZE, m))
        
#         for batchId in range(nrSamplesForEstimation // BATCH_SIZE):
#             print(f"FINISHED BATCH {batchId}/{nrSamplesForEstimation // BATCH_SIZE}")
#             for i in range(BATCH_SIZE):
#                 #if i % 100 == 0:
#                 #    print(f"i = {i}/{nrSamplesForEstimation}")
#                 allSamples = scipy.stats.norm.rvs(loc=0.0, scale=1.0, size = nrAllObservations_n)
#                 all_squared_distances_sorted = getPairwiseSquaredDistanceSorted(allSamples)
#                 inlierSamples = all_squared_distances_sorted[0:m]
                
#                 allExpectations += inlierSamples
                
#                 allSamples_collected[i] = inlierSamples
        
#             S_hat += numpy.cov(allSamples_collected, rowvar = False, bias = True)
        
#         S_hat = S_hat / (nrSamplesForEstimation // BATCH_SIZE)
#         print("GATHERED ALL SAMPLES")
#         print(f"FINISHED COV MATRIX ESTIMATION in {(time.time() - start_time) / 60.0} minutes")
        
#         n = nrSamplesForEstimation
        
#         allExpectations = allExpectations / n
#         S_hat = numpy.diag(1.0 / allExpectations) @ S_hat @ numpy.diag(1.0 / allExpectations)
#         S_hat_squared = S_hat @ S_hat
#         F_hat = (numpy.trace(S_hat) / m) * numpy.eye(m)

#         S_hat_tr_squared = numpy.trace(S_hat) ** 2
#         S_hat_squared_tr = numpy.trace(S_hat_squared)
#         p_RBLW_nominator = ((n - 2) / n) * S_hat_squared_tr + S_hat_tr_squared
#         p_RBLW_denominator = (n + 2) * (S_hat_squared_tr - S_hat_tr_squared / m)
#         p_RBLW = numpy.min([1, p_RBLW_nominator / p_RBLW_denominator])
#         sigmaEstimate = (1 - p_RBLW) * S_hat +  p_RBLW *  F_hat
        
#         print("p_RBLW = ", p_RBLW)
#         print(f"FINISHED COV MATRIX ESTIMATION in {(time.time() - start_time) / 60.0} minutes")
#         start_time = time.time()
#         best_a = optimalLinearCombination(sigmaEstimate)
#         nu = best_a.transpose() @ sigmaEstimate @ best_a
#         print(f"FINISHED INVERSION in {(time.time() - start_time) / 60.0} minutes")
        
#     return allExpectations, best_a, nu


# **** Implements the Rao-Blackwell-Ledoit-Wolf Estimator from "Shrinkage Algorithms for MMSE Covariance Estimation", 2009 ***
def RBLF_estimator(nrAllObservations_n, nrSamplesForEstimation, Qn_estimate = True, m = None):
    if m is not None:
        assert(not Qn_estimate)
    else:
        assert(Qn_estimate)
        m = getQn_m(nrAllObservations_n)

    print("nrAllObservations_n = ", nrAllObservations_n)
    print("nrSamplesForEstimation = ", nrSamplesForEstimation)
    print("m = ", m)
        
    print("USE HOME-MADE COV MATRIX ESTIMATION")
    BATCH_SIZE = 1000
    
    assert((nrSamplesForEstimation % BATCH_SIZE) == 0)

    start_time = time.time()

    allExpectations = numpy.zeros(m)

    S_hat = numpy.zeros((m,m))
    allSamples_collected = numpy.zeros((BATCH_SIZE, m))
    
    for batchId in range(nrSamplesForEstimation // BATCH_SIZE):
        print(f"FINISHED BATCH {batchId}/{nrSamplesForEstimation // BATCH_SIZE}")
        for i in range(BATCH_SIZE):
            allSamples = scipy.stats.norm.rvs(loc=0.0, scale=1.0, size = nrAllObservations_n)
            
            if Qn_estimate:
                all_squared_distances_sorted = getPairwiseSquaredDistanceSorted(allSamples)
                inlierSamples = all_squared_distances_sorted[0:m]
            else:
                squaredSamples = numpy.square(allSamples)
                inlierSamples = numpy.sort(squaredSamples)[0:m]

            allExpectations += inlierSamples
            
            allSamples_collected[i] = inlierSamples
    
        S_hat += numpy.cov(allSamples_collected, rowvar = False, bias = True)
    
    S_hat = S_hat / (nrSamplesForEstimation // BATCH_SIZE)
    print("GATHERED ALL SAMPLES")
    print(f"FINISHED COV MATRIX ESTIMATION in {(time.time() - start_time) / 60.0} minutes")
    
    n = nrSamplesForEstimation
    
    allExpectations = allExpectations / n
    S_hat = numpy.diag(1.0 / allExpectations) @ S_hat @ numpy.diag(1.0 / allExpectations)
    S_hat_squared = S_hat @ S_hat
    F_hat = (numpy.trace(S_hat) / m) * numpy.eye(m)

    S_hat_tr_squared = numpy.trace(S_hat) ** 2
    S_hat_squared_tr = numpy.trace(S_hat_squared)
    p_RBLW_nominator = ((n - 2) / n) * S_hat_squared_tr + S_hat_tr_squared
    p_RBLW_denominator = (n + 2) * (S_hat_squared_tr - S_hat_tr_squared / m)
    p_RBLW = numpy.min([1, p_RBLW_nominator / p_RBLW_denominator])
    sigmaEstimate = (1 - p_RBLW) * S_hat +  p_RBLW *  F_hat
    
    print("p_RBLW = ", p_RBLW)
    print(f"FINISHED COV MATRIX ESTIMATION in {(time.time() - start_time) / 60.0} minutes")
    start_time = time.time()
    best_a = optimalLinearCombination(sigmaEstimate)
    nu = best_a.transpose() @ sigmaEstimate @ best_a
    print(f"FINISHED INVERSION in {(time.time() - start_time) / 60.0} minutes")

    return allExpectations, best_a, nu


# def memEfficientCovMatrix(allSamples_collected):
#     m = allSamples_collected.shape[1]
#     sigmaEstimate = numpy.zeros((m,m))
#     for i in range(allSamples_collected.shape[0]):
#         if i % 100 == 0:
#             print("sigmaEstimate i = ", i)
#         sigmaEstimate += numpy.outer(allSamples_collected[i], allSamples_collected[i])
    
#     return sigmaEstimate
    

def optimalLinearCombination(Sigma):
    m = Sigma.shape[0]

    # if useExact:
    #     invSigma = numpy.linalg.inv(Sigma)
    #     best_a = (invSigma @ numpy.ones(m)) / (  numpy.ones(m).transpose() @ invSigma @ numpy.ones(m) )
    # else:

    L, lower = scipy.linalg.cho_factor(Sigma, overwrite_a=True, check_finite=False)
    inSigma_times_ones = scipy.linalg.cho_solve((L, lower), numpy.ones(m))
    best_a = inSigma_times_ones / (numpy.ones(m) @ inSigma_times_ones)

    return best_a

def getFiniteCorrectedSigmaSquared(inlierAbsDiff, correctionFac_mean, correctionFac_largestInlier):
    # print("correctionFac_mean = ", correctionFac_mean)
    # assert(correctionFac_mean < 1.0)

    uncorrectedSigmaSquared_estimatedWithMean = numpy.mean(numpy.square(inlierAbsDiff))
    uncorrectedSigmaSquared_estimatedWithMax = numpy.max(inlierAbsDiff) ** 2
    
    sigmaSquared_estimatedWithMean = uncorrectedSigmaSquared_estimatedWithMean / correctionFac_mean
    sigmaSquared_estimatedWithMax = uncorrectedSigmaSquared_estimatedWithMax / correctionFac_largestInlier
    
    # uncorrectedSigmaSquared_estimatedWithSmallestVarId = (numpy.sort(inlierAbsDiff)[smallestVarId]) ** 2
    # sigmaSquared_estimatedWithSmallestVarId = uncorrectedSigmaSquared_estimatedWithSmallestVarId / correctionFac_smallestVarId

    return sigmaSquared_estimatedWithMean, sigmaSquared_estimatedWithMax



def getFiniteCorrectedSigmaSquared_bestLinearCombMethod(inlierAbsDiff, allRankExpectations, best_a):

    sortedSquaredInliers = numpy.sort(numpy.square(inlierAbsDiff))
    
    sortedSquaredInliers_normalized = sortedSquaredInliers / allRankExpectations
    
    # print("sortedSquaredInliers = ", sortedSquaredInliers)
    # print("sortedSquaredInliers_normalized = ", sortedSquaredInliers_normalized)
    # print("best_a = ", best_a)

    sigmaSquared_estimatedWithComb = sortedSquaredInliers_normalized @ best_a
    return sigmaSquared_estimatedWithComb

