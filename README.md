
# Improved Variance Estimation from Trimmed Samples

Implementation of all estimators and script for running all experiments described in [Improved Variance Estimation from Trimmed Samples], 2024 (under review).

## Requirements

- Python >= 3.11.2
- R installation
- R package robustbase
- rpy2 >= 3.5.15

## Preparation

1. Install R and R package "robustbase"

2. Create experiment environment using e.g. conda as follows
```bash
conda create -n test python=3.11
conda activate test
```

2. Install necessary packages:
```bash
pip3 install -U rpy2 numpy scikit-learn matplotlib
```

3. Create folders for output using
```bash
mkdir all_results
```

## Usage (Basic Example Workflow)

-------------------------------------------
1. Run Influence Function Simulation
-------------------------------------------
Run influence function simulation with n = 10 and small number of samples (smallRun)
```bash
python influenceFunction_estimate_Qn.py 10 smallRun
```

Run influence function simulation with n = 100 and small number of samples (smallRun)
```bash
python influenceFunction_estimate_Qn.py 100 smallRun
```

All results are saved into folder "all_results/."

-------------------------------------------
2. Show Influence Function
-------------------------------------------

Plots the influence function using the data from the above simulation
```bash
python show_influence_function.py
```

