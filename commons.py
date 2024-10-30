# define constants for the estimators with KNOWN mean

import numpy
from enum import Enum
CorrectionType = Enum("CorrectionType", ["NO_CORRECTION", "ASYMPTOTIC_CORRECTION", "FINITE_CORRECTION_EST_WITH_MEAN", "FINITE_CORRECTION_EST_WITH_MAX", "FINITE_CORRECTION_EST_WITH_COMB", "FINITE_CORRECTION_EST_WITH_COMB_BIASED"])

CORRECTION_TYPES_FOR_TABLES = [CorrectionType.ASYMPTOTIC_CORRECTION, CorrectionType.FINITE_CORRECTION_EST_WITH_MAX, CorrectionType.FINITE_CORRECTION_EST_WITH_MEAN, CorrectionType.FINITE_CORRECTION_EST_WITH_COMB, CorrectionType.FINITE_CORRECTION_EST_WITH_COMB_BIASED]

correctionType_to_label = {}
correctionType_to_label[CorrectionType.NO_CORRECTION] = "$\\hat{\\sigma}^2$"
correctionType_to_label[CorrectionType.ASYMPTOTIC_CORRECTION] = "$\\hat{\\sigma}^2_{\\text{\\normalfont asymptotic}}$"
correctionType_to_label[CorrectionType.FINITE_CORRECTION_EST_WITH_MAX] = r'$\hat{\sigma}^2_{\text{\normalfont finite}}$'
correctionType_to_label[CorrectionType.FINITE_CORRECTION_EST_WITH_MEAN] = r'$\hat{\sigma}^2_{\text{\normalfont mean}}$'
correctionType_to_label[CorrectionType.FINITE_CORRECTION_EST_WITH_COMB] = r'$\hat{\sigma}^2_{\text{\normalfont opt-lin}}$'
correctionType_to_label[CorrectionType.FINITE_CORRECTION_EST_WITH_COMB_BIASED] = r'$\hat{\sigma}^2_{\text{\normalfont opt-mse}}$'

def get_sample_size(smallRun):
    if smallRun:
        # USE THIS FOR DEBUG ONLY:
        nr_runs = 1000
        nr_mc_samples_for_sigma_estimate = 500000
    else:
        nr_runs = 100000
        nr_mc_samples_for_sigma_estimate = 500000

    return nr_runs, nr_mc_samples_for_sigma_estimate


def getAll_m(n):
    if n == 10:
        all_m = numpy.asarray([2, 5, 7, 9]) 
    else:
        all_m = numpy.asarray([20, 50, 70, 90, 99])
        all_m = all_m * (n / 100)
        all_m = all_m.astype(int)
    
    return all_m

def getNiceLatexExp(eNotationStr, highlight = False):
    firstPart = eNotationStr.split("e")[0]
    secondPart = int(eNotationStr.split("e")[1])

    if highlight:
        firstPart = r"\textbf{" + firstPart + "}"

    if secondPart == 0:
        return firstPart
    else:
        if highlight:
            secondPart = r" $\boldsymbol{\times 10^{" + str(secondPart) + "}}$"
        else:
            secondPart = r" $\times 10^{" + str(secondPart) + "}$"
        return firstPart + secondPart