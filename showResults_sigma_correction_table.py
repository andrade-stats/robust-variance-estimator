# used to create all tables with detailed evaluation of bias, variance and MSE of all estimators

import numpy
from commons import getNiceLatexExp

import commons
from commons import CORRECTION_TYPES_FOR_TABLES
from commons import correctionType_to_label


NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 500000
ROUND_DIGITS = 2


def getStrSummary(results_allRuns, highlight = False):
    meanVal = numpy.mean(results_allRuns)
    mce = numpy.std(results_allRuns)
    stdResultStr = f" ({ getNiceLatexExp('{:0.1e}'.format(mce)) }) "
    if highlight:
        meanResultStr = r"\textbf{" + str(round(meanVal, ROUND_DIGITS)) + "}"
    else:
        meanResultStr = f"{round(meanVal, ROUND_DIGITS)}"
    return meanResultStr + stdResultStr


def getRoundedMean(results_allRuns):
    meanVal = numpy.mean(results_allRuns)
    return round(meanVal, ROUND_DIGITS)

def getBest_methods(type, n, m = None):

    if SIGMA_SQUARE_ESTIMATION_WITH_GIVEN_MEAN:
        allResults = numpy.load("all_results/" + f"allResults_{type}_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()
    else:
        allResults = numpy.load("all_results/" + f"Qn_allResults_{type}_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()

    allMethods = numpy.zeros(len(CORRECTION_TYPES_FOR_TABLES))

    for rowId, correctionType in enumerate(CORRECTION_TYPES_FOR_TABLES):
        allMethods[rowId] = getRoundedMean(allResults[correctionType])

    minIds = numpy.where(allMethods == numpy.min(numpy.abs(allMethods)))[0]
    return numpy.asarray(CORRECTION_TYPES_FOR_TABLES)[minIds]


SIGMA_SQUARE_ESTIMATION_WITH_GIVEN_MEAN = True

# vspace = "[0.2cm] "
vspace = ""

print(" ")
print(" ")
print(" ")

if SIGMA_SQUARE_ESTIMATION_WITH_GIVEN_MEAN:
    n = 10
    
    ALL_M = commons.getAll_m(n)
    
    for m in ALL_M:

        bestMSEMethods = getBest_methods("mse", n, m)
        bestVarMethods = getBest_methods("var", n, m)
        bestBiasMethods = getBest_methods("bias", n, m)

        print(r'\midrule')

        allResults_bias = numpy.load("all_results/" + f"allResults_bias_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()
        allResults_var = numpy.load("all_results/" + f"allResults_var_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()
        allResults_mse = numpy.load("all_results/" + f"allResults_mse_{n}_{m}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()

        for rowId, correctionType in enumerate(CORRECTION_TYPES_FOR_TABLES):
            
            if rowId == 0:
                prefix = r'\multirow{5}{*}{$m = ' + str(m) + r'$}'
            else:
                prefix = ""

            print(prefix + f" & {correctionType_to_label[correctionType]} & {getStrSummary(allResults_bias[correctionType], correctionType in bestBiasMethods)} & {getStrSummary(allResults_var[correctionType],  correctionType in bestVarMethods)} & {getStrSummary(allResults_mse[correctionType],  correctionType in bestMSEMethods)} \\\\" + vspace)
            
else:

    n = 10

    from commons_Qn import CORRECTION_TYPES_FOR_TABLES
    from commons_Qn import correctionType_to_label

    bestMSEMethods = getBest_methods("mse", n)
    bestVarMethods = getBest_methods("var", n)
    bestBiasMethods = getBest_methods("bias", n)

    allResults_bias = numpy.load("all_results/" + f"Qn_allResults_bias_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()
    allResults_var = numpy.load("all_results/" + f"Qn_allResults_var_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()
    allResults_mse = numpy.load("all_results/" + f"Qn_allResults_mse_{n}_MCSamples{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}" + ".npy", allow_pickle=True).item()

    for rowId, correctionType in enumerate(CORRECTION_TYPES_FOR_TABLES):
        
        if rowId == 0:
            prefix = r'\multirow{5}{*}{$n = ' + str(n) + r'$}'
        else:
            prefix = ""
        
        print(prefix + f" & {correctionType_to_label[correctionType]} & {getStrSummary(allResults_bias[correctionType], correctionType in bestBiasMethods)} & {getStrSummary(allResults_var[correctionType], correctionType in bestVarMethods)} & {getStrSummary(allResults_mse[correctionType], correctionType in bestMSEMethods)} \\\\"  + vspace)

