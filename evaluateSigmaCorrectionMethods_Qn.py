# evaluation of proposed variance estimtor with UNKNOWN mean for data WITHOUT outliers

# Example usage:
# python evaluateSigmaCorrectionMethods_Qn.py 10
# saves results in folder "all_results"

from statistics import variance
import scipy.stats
import numpy
import scipy.special
import time
import sigmaCorrectionMethods
from commons_Qn import CorrectionType
import sys

import rpy2
print(rpy2.__version__)

from rpy2.robjects.packages import importr
robustbase = importr('robustbase') # import R's "robustbase" package

from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

# Create a converter that starts with rpy2's default converter
# to which the numpy conversion rules are added.
np_cv_rules = default_converter + numpy2ri.converter


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
    n = 10
else:
    assert(len(sys.argv) == 2)
    n = int(sys.argv[1])

USE_MEMORY_EFFICIENT = True

assert(n >= 10 and n <= 500)

m = sigmaCorrectionMethods.getQn_m(n)

trueVar = 10.0

print("n = ", n)
print("NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = ", NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE)

for mceRunId in range(MONTE_CARLO_ERROR_ESTIMATION_RUNS):
    print(f"******* Q_n MCE RUN {mceRunId} (n = {n}) *********")

    start_time = time.time()

    allRankExpectations, best_a, nu = sigmaCorrectionMethods.RBLF_estimator(n, nrSamplesForEstimation = NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE, Qn_estimate = True)
    # allRankExpectations, best_a, nu = sigmaCorrectionMethods.estimateFiniteCorrectionFactors_Qn(n, nrSamplesForEstimation = NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE, useMemoryEfficient = USE_MEMORY_EFFICIENT)
    
    print("-- FINISHED PREPARATION --")

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
        allSamples = scipy.stats.norm.rvs(loc=0.0, scale=numpy.sqrt(trueVar), size = n)
        
        with localconverter(np_cv_rules) as cv:
            
            estimate_allMethods[CorrectionType.MAD][i] = rpy2.robjects.r.mad(allSamples) ** 2
            estimate_allMethods[CorrectionType.QN_ASYMPTOTIC][i] = robustbase.Qn(allSamples, finite_corr = False) ** 2
            estimate_allMethods[CorrectionType.QN_FINITE][i] = robustbase.Qn(allSamples, finite_corr = True) ** 2

        all_squared_distances_sorted = sigmaCorrectionMethods.getPairwiseSquaredDistanceSorted(allSamples)
        all_inlier_distances = numpy.sqrt(all_squared_distances_sorted[0:m])

        sigmaSquared_lowestVar_unbiased_estimator = sigmaCorrectionMethods.getFiniteCorrectedSigmaSquared_bestLinearCombMethod(all_inlier_distances, allRankExpectations, best_a)
        estimate_allMethods[CorrectionType.QN_OPT_LIN][i] = sigmaSquared_lowestVar_unbiased_estimator

        shrinkageFac = 1.0 / (1.0 + nu)
        estimate_allMethods[CorrectionType.QN_OPT_MSE][i] = shrinkageFac * sigmaSquared_lowestVar_unbiased_estimator
        
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
numpy.save("all_results/" + f"Qn_allResults_bias_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", allResults_bias)
numpy.save("all_results/" + f"Qn_allResults_var_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", allResults_var)
numpy.save("all_results/" + f"Qn_allResults_mse_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}", allResults_mse)


def showSummary(s):
    return str(numpy.mean(s)) + " " + str(numpy.std(s))

print("MEAN AND MCE ESTIMATES: ")
for rowId, correctionType in enumerate(CorrectionType):
    print("correctionType = ", correctionType)
    print("bias = " + showSummary(allResults_bias[correctionType]))
    print("var = " + showSummary(allResults_var[correctionType]))
    print("mse = " + showSummary(allResults_mse[correctionType]))

print("FINISHED ALL MCE RUNS")
print("n = ", n)
print("NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = ", NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE)
print("USE_MEMORY_EFFICIENT = ", USE_MEMORY_EFFICIENT)