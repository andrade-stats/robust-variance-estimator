
import numpy
from commons_Qn import CorrectionType
from commons_Qn import correctionType_to_label
from commons import getNiceLatexExp
from commons import get_sample_size

ROUND_DIGITS = 1


def getStrSummary(results_allRuns, highlight = False):
    meanVal = numpy.mean(results_allRuns)
    mce = numpy.std(results_allRuns)
    stdResultStr = f" ({ getNiceLatexExp('{:0.1e}'.format(mce)) }) "
    meanResultStr = f"{getNiceLatexExp('{:0.1e}'.format(meanVal), highlight)}"
    return meanResultStr + stdResultStr

NR_RUNS, NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = get_sample_size(smallRun = True)

n = 10
# n = 100

filename_stem = "all_results/" + f"influenceFunction_{NR_RUNS}_{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}_{n}"
all_runtimes_raw = numpy.load(filename_stem + "_" + "runtimes" + ".npy", allow_pickle = True).item()

print("RBLF_estimator (Seconds) = ", all_runtimes_raw["RBLF_estimator"])
print("RBLF_estimator (Minutes) = ", all_runtimes_raw["RBLF_estimator"] / 60.0)

CORRECTION_TYPES_FOR_TABLES = [CorrectionType.MAD, CorrectionType.QN_ASYMPTOTIC, CorrectionType.QN_FINITE, CorrectionType.QN_OPT_LIN, CorrectionType.QN_OPT_MSE]


all_runtimes = numpy.zeros((len(CORRECTION_TYPES_FOR_TABLES), all_runtimes_raw["getPairwiseSquaredDistanceSorted"].flatten().shape[0]))

for rowId, correctionType in enumerate(CORRECTION_TYPES_FOR_TABLES):
    if correctionType == CorrectionType.QN_OPT_LIN or correctionType == CorrectionType.QN_OPT_MSE:
        all_runtimes[rowId] = all_runtimes_raw[correctionType].flatten() + all_runtimes_raw["getPairwiseSquaredDistanceSorted"].flatten()
    else:
        all_runtimes[rowId] = all_runtimes_raw[correctionType].flatten() 

rounded_values = numpy.mean(all_runtimes, axis = 1) 
best_value = numpy.min(rounded_values)
best_ids = set(numpy.where(rounded_values == best_value)[0])


for rowId, correctionType in enumerate(CORRECTION_TYPES_FOR_TABLES):
    
    prefix = " & "
    
    highlight = (rowId in best_ids)

    if correctionType == CorrectionType.QN_OPT_LIN or correctionType == CorrectionType.QN_OPT_MSE:
        
        label = correctionType_to_label[correctionType]
        print(prefix + f" {label} & {getStrSummary(all_runtimes[rowId], highlight)} \\\\" )
    
    else:
        print(prefix + f" {correctionType_to_label[correctionType]} & {getStrSummary(all_runtimes[rowId], highlight)} \\\\" )
    
