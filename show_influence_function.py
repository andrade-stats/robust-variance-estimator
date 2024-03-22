# used to draw the influence function of the proposed estimator QN_OPT_MSE (= Q^2_{opt-mse}) 

import sys
import numpy
from commons_Qn import CorrectionType
from commons_Qn import correctionType_to_label

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}" # for \text command


SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


# set colors 
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

correction_type_to_color = {}
for i, correctionType in enumerate(CorrectionType):
    correction_type_to_color[correctionType] = colors[i]

tmp = correction_type_to_color[CorrectionType.QN_OPT_LIN]
correction_type_to_color[CorrectionType.QN_OPT_LIN] = correction_type_to_color[CorrectionType.QN_OPT_MSE]
correction_type_to_color[CorrectionType.QN_OPT_MSE] = tmp




def plot(ax, all_outlier_values, mseEstimate, correctionType):
    neg_ids = numpy.argsort(-all_outlier_values)
    x_all_outlier_values = numpy.append(- all_outlier_values[neg_ids], all_outlier_values)
    y_all_mse = numpy.append(mseEstimate[neg_ids], mseEstimate)
    ax.plot(x_all_outlier_values, y_all_mse, linewidth=2.0, label = correctionType_to_label[correctionType], color = correction_type_to_color[correctionType])
    

NR_RUNS = 100000 
NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 500000 

if len(sys.argv) == 1:
    smallRun = False
else:
    assert(len(sys.argv) == 2)
    assert(sys.argv[1] == "smallRun")
    smallRun = True


if smallRun:
    # USE THIS FOR DEBUG ONLY:
    NR_RUNS = 100
    NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 50000
else:
    NR_RUNS = 100000
    NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = 500000




# -------------- n = 10 --------------
    
n = 10

filename_stem = "all_results/" + f"influenceFunction_{NR_RUNS}_{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}_{n}"
all_results_mse_n10 = numpy.load(filename_stem + "_mse" + ".npy", allow_pickle = True).item()
all_outlier_values = numpy.load(filename_stem + "_steps" + ".npy")

fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(9,15), height_ratios=[1, 1, 2])

ax2.set_ylim(0, 180)  
ax.set_ylim(200, 1000) # outliers only

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.set_title(r"$n = 10$", fontsize=MEDIUM_SIZE)



# -------------- n = 100 --------------
    
n = 100

filename_stem = "all_results/" + f"influenceFunction_{NR_RUNS}_{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}_{n}"
all_results_mse_n100 = numpy.load(filename_stem + "_mse" + ".npy", allow_pickle = True).item()

ax3.set_title(r"$n = 100$", fontsize=MEDIUM_SIZE)

for correctionType in CorrectionType:
    print("*************************")
    print("method = ", correctionType)

    plot(ax3, all_outlier_values, all_results_mse_n100[correctionType], correctionType)

    if correctionType != CorrectionType.QN_ASYMPTOTIC:
        plot(ax2, all_outlier_values, all_results_mse_n10[correctionType], correctionType)
    else:
        plot(ax, all_outlier_values, all_results_mse_n10[correctionType], correctionType)


ax2.set_ylabel(r"$\text{MSE}(T, D_n)$", loc="top",  fontsize=MEDIUM_SIZE)
ax3.set_ylabel(r"$\text{MSE}(T, D_n)$",  fontsize=MEDIUM_SIZE) 
ax3.set_xlabel("x",  fontsize=MEDIUM_SIZE)


handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol = 5)

plt.show()
# plt.tight_layout()
# plt.savefig("latex/plots/" + f"influence_function_mse_n10_n100" + ".pdf")


