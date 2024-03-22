# evaluation of proposed variance estimtor with UNKNOWN mean for data WITH outlier

import scipy.stats
import numpy
import scipy.special
import time
import sigmaCorrectionMethods
from commons_Qn import CorrectionType
import sys

import rpy2
print("rpy2 - Version = ", rpy2.__version__)

from rpy2.robjects.packages import importr
robustbase = importr('robustbase') # import R's "robustbase" package

from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

# Create a converter that starts with rpy2's default converter
# to which the numpy conversion rules are added.
np_cv_rules = default_converter + numpy2ri.converter


numpy.random.seed(3523421)


if len(sys.argv) == 1:
    n = 10
    smallRun = False
else:
    assert(len(sys.argv) <= 3)
    n = int(sys.argv[1])

    if len(sys.argv) == 3:
        assert(sys.argv[2] == "smallRun")
        smallRun = True
    else:
        smallRun = False

if smallRun:
    # USE THIS FOR DEBUG ONLY:
    NR_RUNS = 1000
    NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 50000
else:
    NR_RUNS = 100000
    NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 500000


USE_MEMORY_EFFICIENT = True

assert(n >= 10 and n <= 500)

NR_OUTLIER_STEPS = 500

m = sigmaCorrectionMethods.getQn_m(n)

trueVar = 10.0

MAX_OUTLIER_VALUE = numpy.sqrt(trueVar) * 5

print("n = ", n)
print("NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = ", NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE)


start_time = time.time()

allRankExpectations, best_a, nu = sigmaCorrectionMethods.RBLF_estimator(n, nrSamplesForEstimation = NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE, Qn_estimate = True)
# allRankExpectations, best_a, nu = sigmaCorrectionMethods.estimateFiniteCorrectionFactors_Qn(n, nrSamplesForEstimation = NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE, useMemoryEfficient = USE_MEMORY_EFFICIENT)

print("-- FINISHED PREPARATION --")

variance_allMethods = {}
estimate_allMethods = {}
mse_allMethods = {}
for correctionType in CorrectionType:
    estimate_allMethods[correctionType] = numpy.zeros((NR_RUNS, NR_OUTLIER_STEPS))
    variance_allMethods[correctionType] = numpy.zeros((NR_RUNS, NR_OUTLIER_STEPS))
    mse_allMethods[correctionType] = numpy.zeros((NR_RUNS, NR_OUTLIER_STEPS))

print("*******************")
print("START RUNS")
print("*******************")

all_outlier_values = numpy.linspace(start = 0.0, stop = MAX_OUTLIER_VALUE, num = NR_OUTLIER_STEPS)
assert(all_outlier_values.shape[0] == NR_OUTLIER_STEPS)

for i in range(NR_RUNS):
    allSamples = scipy.stats.norm.rvs(loc=0.0, scale=numpy.sqrt(trueVar), size = n)
    
    print("run = ", i)
    
    for j in range(NR_OUTLIER_STEPS):
        
        # outlier
        allSamples[0] = all_outlier_values[j]

        with localconverter(np_cv_rules) as cv:
            
            estimate_allMethods[CorrectionType.MAD][i, j] = rpy2.robjects.r.mad(allSamples) ** 2
            estimate_allMethods[CorrectionType.QN_ASYMPTOTIC][i, j] = robustbase.Qn(allSamples, finite_corr = False) ** 2
            estimate_allMethods[CorrectionType.QN_FINITE][i, j] = robustbase.Qn(allSamples, finite_corr = True) ** 2

        all_squared_distances_sorted = sigmaCorrectionMethods.getPairwiseSquaredDistanceSorted(allSamples)
        all_inlier_distances = numpy.sqrt(all_squared_distances_sorted[0:m])

        sigmaSquared_lowestVar_unbiased_estimator = sigmaCorrectionMethods.getFiniteCorrectedSigmaSquared_bestLinearCombMethod(all_inlier_distances, allRankExpectations, best_a)
        estimate_allMethods[CorrectionType.QN_OPT_LIN][i, j] = sigmaSquared_lowestVar_unbiased_estimator

        shrinkageFac = 1.0 / (1.0 + nu)
        estimate_allMethods[CorrectionType.QN_OPT_MSE][i, j] = shrinkageFac * sigmaSquared_lowestVar_unbiased_estimator


all_results_mse = {}

print("*************************")
for correctionType in CorrectionType:
    print("*************************")
    print("method = ", correctionType)
    mseEstimate = numpy.mean(numpy.square(estimate_allMethods[correctionType] - trueVar), axis = 0)
    all_results_mse[correctionType] = mseEstimate

filename_stem = "all_results/" + f"influenceFunction_{NR_RUNS}_{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}_{n}"
numpy.save(filename_stem + "_mse", all_results_mse)
numpy.save(filename_stem + "_steps", all_outlier_values)