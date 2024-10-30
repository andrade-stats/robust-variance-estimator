# used to draw the influence function of the proposed estimator QN_OPT_MSE (= Q^2_{opt-mse}) 

import sys
import numpy
from commons_Qn import CorrectionType
from commons_Qn import correctionType_to_label
from commons import get_sample_size

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}" # for \text command

def load_file(type, n):
    filename_stem = "all_results/" + f"influenceFunction_{NR_RUNS}_{NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE}_{n}"
    if type == "steps":
        return numpy.load(filename_stem + "_" + type + ".npy")
    else:
        return numpy.load(filename_stem + "_" + type + ".npy", allow_pickle = True).item()





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
    

NR_RUNS, NR_MC_SAMPLES_FOR_SIGMA_ESTIMATE = get_sample_size(smallRun = False)


# show = "MSE"
show = "BIAS_AND_VAR"


all_outlier_values_n10 = load_file(type = "steps", n = 10)
all_outlier_values_n100 = load_file(type = "steps", n = 100)

if show == "MSE":
    all_results_n10 = load_file(type = "mse", n=10)
    all_results_n100 = load_file(type = "mse", n=100)

else:
    all_results_n10_bias = load_file(type = "bias", n=10)
    all_results_n100_bias = load_file(type = "bias", n=100)
    
    all_results_n10_var = load_file(type = "var", n=10)
    all_results_n100_var = load_file(type = "var", n=100)


if show == "BIAS_AND_VAR":
    fig, all_axes = plt.subplots(2, 2, sharex=True, figsize=(12,15)) # , height_ratios=[1, 1])
    all_axes[0,0].set_title(r"$n = 10$", fontsize=MEDIUM_SIZE)
    all_axes[0,1].set_title(r"$n = 10$", fontsize=MEDIUM_SIZE)

    all_axes[1,0].set_title(r"$n = 100$", fontsize=MEDIUM_SIZE)
    all_axes[1,1].set_title(r"$n = 100$", fontsize=MEDIUM_SIZE)
else:    
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


for correctionType in CorrectionType:
    print("*************************")
    print("method = ", correctionType)

    if show == "BIAS_AND_VAR":
        plot(all_axes[0,0], all_outlier_values_n10, all_results_n10_bias[correctionType], correctionType)
        plot(all_axes[0,1], all_outlier_values_n10, all_results_n10_var[correctionType], correctionType)
        plot(all_axes[1,0], all_outlier_values_n100, all_results_n100_bias[correctionType], correctionType)
        plot(all_axes[1,1], all_outlier_values_n100, all_results_n100_var[correctionType], correctionType)
    else:
        ax3.set_title(r"$n = 100$", fontsize=MEDIUM_SIZE)
        plot(ax3, all_outlier_values_n100, all_results_n100[correctionType], correctionType)
        if correctionType != CorrectionType.QN_ASYMPTOTIC:
            plot(ax2, all_outlier_values_n10, all_results_n10[correctionType], correctionType)
        else:
            plot(ax, all_outlier_values_n10, all_results_n10[correctionType], correctionType)


if show == "BIAS_AND_VAR":
    all_axes[0,0].set_ylabel(r"$\text{" + "BIAS" + r"}(T, D_n)$",  fontsize=MEDIUM_SIZE)
    all_axes[0,1].set_ylabel(r"$\text{" + "VAR" + r"}(T, D_n)$",  fontsize=MEDIUM_SIZE)
    all_axes[1,0].set_ylabel(r"$\text{" + "BIAS" + r"}(T, D_n)$",  fontsize=MEDIUM_SIZE)
    all_axes[1,1].set_ylabel(r"$\text{" + "VAR" + r"}(T, D_n)$",  fontsize=MEDIUM_SIZE)
    handles, labels = all_axes[0,0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.1)
    plt.subplots_adjust(wspace=0.25)
    fig.legend(handles, labels, loc='lower center', ncol = 5) #  bbox_to_anchor=(0.0,-0.5))
    # plt.tight_layout(pad = 1.5, h_pad = 5.0)

else:
    ax2.set_ylabel(r"$\text{" + show + r"}(T, D_n)$", loc="top",  fontsize=MEDIUM_SIZE)

    ax3.set_ylabel(r"$\text{" + show + r"}(T, D_n)$",  fontsize=MEDIUM_SIZE) 
    ax3.set_xlabel("x",  fontsize=MEDIUM_SIZE)
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol = 5)
    plt.tight_layout()

# plt.show()
plt.savefig("latex/plots/" + f"influence_function_" + show.lower() + "_n10_n100" + ".pdf")


