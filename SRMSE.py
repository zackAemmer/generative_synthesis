import numpy as np
import pandas as pd

def calculateSRMSE(synthetic_pop, true_pop):

    # Both populations should have exact same columns in exact same order
    assert list(synthetic_pop.columns) == list(true_pop.columns)

    mse_vals = []
    mf_vals = []
    for col in synthetic_pop.columns:

        # Get bin frequencies for each column
        synthetic_freqs = synthetic_pop[col].value_counts().to_dict()
        true_freqs = true_pop[col].value_counts().to_dict()

        # Calculate squared error for each bin; keep track of mean frequencies
        se_vals = []
        freq_vals = []
        for col_bin in list(true_freqs.keys()):
            # There may not be counts of certain bins in the synthetic population
            if col_bin in synthetic_freqs.keys():
                se = (synthetic_freqs[col_bin] - true_freqs[col_bin])**2 / len(true_freqs.keys())
            else:
                se = (0 - true_freqs[col_bin])**2 / len(true_freqs.keys())
            se_vals.append(se)
            freq_vals.append(true_freqs[col_bin] / len(true_freqs.keys()))

        mse_vals.append(np.sum(se_vals))
        mf_vals.append(np.sum(freq_vals))

    # Reduce squared errors to RMSE for each variable
    rmse = np.sum(mse_vals)**.5
    mf = np.sum(mf_vals)
    srmse = rmse / mf
    print(f"Univariate (marginal) SRMSE: {srmse}")
    return srmse

def calculateBivariateSRMSE(synthetic_pop, true_pop):

    # Both populations should have exact same columns in exact same order
    assert list(synthetic_pop.columns) == list(true_pop.columns)

    mse_vals = []
    mf_vals = []
    used_combos = []
    # Create contingency table for every combination of 2 variables
    for col_1 in list(synthetic_pop.columns):
        for col_2 in list(synthetic_pop.columns):

            # Don't do contingency of the same column on itself or repeat tables
            if col_1 == col_2:
                continue
            elif [col_2,col_1] in used_combos or [col_1,col_2] in used_combos:
                continue
            else:
                ct_synth = pd.crosstab(synthetic_pop[col_1], synthetic_pop[col_2], margins=False)
                ct_true = pd.crosstab(true_pop[col_1], true_pop[col_2], margins=False)
                used_combos.append([col_1, col_2])
            
            # Calculate MSE on the contingency table
            z = ((ct_synth - ct_true)**2).values

            # There may not be counts of certain bins in the synthetic population
            nan_indices = np.argwhere(np.isnan(z))
            nan_indices = [tuple(idx) for idx in nan_indices]
            for idx in nan_indices:
                z[idx] = ct_true.values[idx]**2
            
            mse_vals.append(np.sum(z) / z.shape[0]*z.shape[1])
            mf_vals.append(np.sum(ct_true.values) / z.shape[0]*z.shape[1])
    srmse = np.sum(mse_vals)**.5 / np.sum(mf_vals)
    print(f"Bivariate (joint) SRMSE: {srmse}")
    return srmse