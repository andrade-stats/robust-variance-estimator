# evaluation of proposed variance estimtor with KNOWN mean for data WITHOUT outliers

# Example usage
# python evaluateSigmaCorrectionMethods.py 10
# saves results in folder "all_results"

import scipy.stats
import numpy
import scipy.special
import time
import sigmaCorrectionMethods
import commons
from commons import CorrectionType
import sys

def getBestSubsetSamples(stdCandidate, n, m):
    allSamples = scipy.stats.norm.rvs(loc=0.0, scale=stdCandidate, size = n)
    allSamplesLogProb = scipy.stats.norm.logpdf(allSamples, loc=0.0, scale = stdCandidate)
    bestIds = numpy.argsort(-allSamplesLogProb)[0:m]
    bestSamplesInOrder = allSamples[bestIds]
    others = numpy.delete(allSamples, bestIds)
    return bestSamplesInOrder, others


def getBestRepeatedSamples(stdCandidate, n, m):
    allCollectedSamples = numpy.zeros(m)
    
    bestOfNrSamples = n - m + 1
    
    for i in range(m):
        allSamples = scipy.stats.norm.rvs(loc=0.0, scale=stdCandidate, size = bestOfNrSamples)
        allSamplesLogProb = scipy.stats.norm.logpdf(allSamples, loc=0.0, scale = stdCandidate)
        newSample = allSamples[numpy.argmax(allSamplesLogProb)]
        allCollectedSamples[i] = newSample
    
    return allCollectedSamples


numpy.random.seed(3523421)


MONTE_CARLO_ERROR_ESTIMATION_RUNS = 100
NR_RUNS = 100000
NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 500000

allResults_bias = {}
allResults_var = {}
allResults_mse = {}
for t in CorrectionType:
    allResults_bias[t] = numpy.zeros(MONTE_CARLO_ERROR_ESTIMATION_RUNS)
    allResults_var[t] = numpy.zeros(MONTE_CARLO_ERROR_ESTIMATION_RUNS)
    allResults_mse[t] = numpy.zeros(MONTE_CARLO_ERROR_ESTIMATION_RUNS)


if len(sys.argv) == 1:
    n = 100
else:
    assert(len(sys.argv) == 2)
    n = int(sys.argv[1])
    

assert(n >= 10 and n <= 10000)
print("n = ", n)

ALL_M = commons.getAll_m(n)



print("ALL_M = ", ALL_M)

for m in ALL_M:

    trueVar = 10.0

    all_best_a = numpy.zeros((MONTE_CARLO_ERROR_ESTIMATION_RUNS, m))

    for mceRunId in range(MONTE_CARLO_ERROR_ESTIMATION_RUNS):
        print(f"******* MCE RUN {mceRunId} (m = {m}) *********")

        start_time = time.time()

        allRankExpectations, best_a, nu = sigmaCorrectionMethods.RBLF_estimator(n, nrSamplesForEstimation = NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE, Qn_estimate = False, m = m)
        
        assert(allRankExpectations.shape[0] == m)
        correctionFac_largestInlier = allRankExpectations[m-1]
        correctionFac_mean = numpy.mean(allRankExpectations)

        # correctionFac_mean, correctionFac_largestInlier, smallestVarId, correctionFac_smallestVarId, allRankExpectations, best_a, nu = sigmaCorrectionMethods.estimateFiniteCorrectionFactors(m, n, nrSamplesForEstimation = NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE)
        all_best_a[mceRunId] = best_a

        variance_allMethods = {}
        estimate_allMethods = {}
        mse_allMethods = {}
        for correctionType in CorrectionType:
            estimate_allMethods[correctionType] = numpy.zeros(NR_RUNS)
            variance_allMethods[correctionType] = numpy.zeros(NR_RUNS)
            mse_allMethods[correctionType] = numpy.zeros(NR_RUNS)

        print("*******************")
        print("START RUNS")
        print("*******************")

        for i in range(NR_RUNS):
            bestSamplesInOrder, _ = getBestSubsetSamples(numpy.sqrt(trueVar), n, m)
            assert(bestSamplesInOrder.shape[0] == m)
            inlierAbsDiff = numpy.abs(bestSamplesInOrder)

            estimate_allMethods[CorrectionType.NO_CORRECTION][i] = numpy.sum(numpy.square(inlierAbsDiff)) / m
            
            asymptoticCorrectedSigmaSquare = sigmaCorrectionMethods.getAsymptoticCorrectedSigma(inlierAbsDiff, n) ** 2
            # print("correctedSigmaSquare (asymptotic) = ", asymptoticCorrectedSigmaSquare)
            estimate_allMethods[CorrectionType.ASYMPTOTIC_CORRECTION][i] = asymptoticCorrectedSigmaSquare
            
            finiteCorrectedSigmaSquare_estimatedWithMean, finiteCorrectedSigmaSquare_estimatedWithMax = sigmaCorrectionMethods.getFiniteCorrectedSigmaSquared(inlierAbsDiff, correctionFac_mean, correctionFac_largestInlier)
            estimate_allMethods[CorrectionType.FINITE_CORRECTION_EST_WITH_MEAN][i] = finiteCorrectedSigmaSquare_estimatedWithMean
            estimate_allMethods[CorrectionType.FINITE_CORRECTION_EST_WITH_MAX][i] = finiteCorrectedSigmaSquare_estimatedWithMax
            # estimate_allMethods[CorrectionType.FINITE_CORRECTION_EST_WITH_SMALLEST_VAR_ID][i] = finiteCorrectedSigmaSquare_estimatedWithSmallestVarId

            assert(inlierAbsDiff.shape[0] == m)
            assert(inlierAbsDiff.shape[0] == best_a.shape[0])
            sigmaSquared_lowestVar_unbiased_estimator = sigmaCorrectionMethods.getFiniteCorrectedSigmaSquared_bestLinearCombMethod(inlierAbsDiff, allRankExpectations, best_a)
            estimate_allMethods[CorrectionType.FINITE_CORRECTION_EST_WITH_COMB][i] = sigmaSquared_lowestVar_unbiased_estimator

            shrinkageFac = 1.0 / (1.0 + nu)
            estimate_allMethods[CorrectionType.FINITE_CORRECTION_EST_WITH_COMB_BIASED][i] = shrinkageFac * sigmaSquared_lowestVar_unbiased_estimator
            
        print("*************************")
        for correctionType in CorrectionType:
            print("*************************")
            print("method = ", correctionType)
            biasEstimate = numpy.mean(estimate_allMethods[correctionType] - trueVar)
            varEstimate = numpy.var(estimate_allMethods[correctionType])
            mseEstimate = numpy.mean(numpy.square(estimate_allMethods[correctionType] - trueVar))
            print("bias = ", biasEstimate)
            print("variance = ", varEstimate)
            print("MSE = ", mseEstimate)


        ROUND_DIGITS = 2

        print("**** for latex *******")
        print(r"\midrule")
        for rowId, correctionType in enumerate(CorrectionType):
            biasEstimate = numpy.mean(estimate_allMethods[correctionType] - trueVar)
            varEstimate = numpy.var(estimate_allMethods[correctionType], ddof = 1)
            mseEstimate = numpy.mean(numpy.square(estimate_allMethods[correctionType] - trueVar))

            allResults_bias[correctionType][mceRunId] = biasEstimate
            allResults_var[correctionType][mceRunId] = varEstimate
            allResults_mse[correctionType][mceRunId] = mseEstimate

        print(f"FINISHED ONE MCE RUN IN {(time.time() - start_time) / 60.0} minutes")


    print("**** SAVE ALL RESULTS ****")
    numpy.save("all_results/" + f"all_best_a_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", all_best_a)
    numpy.save("all_results/" + f"allResults_bias_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", allResults_bias)
    numpy.save("all_results/" + f"allResults_var_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", allResults_var)
    numpy.save("all_results/" + f"allResults_mse_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", allResults_mse)


    def showSummary(s):
        return str(numpy.mean(s)) + " " + str(numpy.std(s))

    print("MEAN AND MCE ESTIMATES: ")
    for rowId, correctionType in enumerate(CorrectionType):
        print("correctionType = ", correctionType)
        print("bias = " + showSummary(allResults_bias[correctionType]))
        print("var = " + showSummary(allResults_var[correctionType]))
        print("mse = " + showSummary(allResults_mse[correctionType]))

    print("FINISHED ALL MCE RUNS")
    print("NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = ", NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE)

