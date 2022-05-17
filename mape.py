import numpy as np
import pandas as pd

def calculateMAPE(synthetic_pop, true_pop):

    # Both populations should have exact same columns in exact same order
    assert list(synthetic_pop.columns) == list(true_pop.columns)

    ape_vals_all = []
    ape_vals_ind = []
    total_bins = 0
    for col in synthetic_pop.columns:

        # Get bin frequencies for each column
        synthetic_freqs = synthetic_pop[col].value_counts().to_dict()
        true_freqs = true_pop[col].value_counts().to_dict()

        # Calculate squared error for each bin; keep track of mean frequencies
        pe_vals = []
        for col_bin in list(true_freqs.keys()):
            # There may not be counts of certain bins in the synthetic population
            if col_bin in synthetic_freqs.keys():
                pe = (true_freqs[col_bin] - synthetic_freqs[col_bin]) / true_freqs[col_bin]
            else:
                pe = (true_freqs[col_bin] - 0) / true_freqs[col_bin]
            pe_vals.append(np.absolute(pe))

        ape_vals_all.append(np.sum(pe_vals))
        ape_vals_ind.append(np.sum(pe_vals) / len(true_freqs.keys())*100)
        total_bins += len(true_freqs.keys())
        print(f"{col} MAPE: {np.sum(pe_vals) / len(true_freqs.keys())*100}")

    # Reduce to mean for all variables
    mape = np.sum(ape_vals_all) / total_bins * 100
    print(f"Univariate (marginal) MAPE: {mape}, Total Bins: {total_bins}")
    return ape_vals_ind

# def calculateBivariateMAPE(synthetic_pop, true_pop):

#     # Both populations should have exact same columns in exact same order
#     assert list(synthetic_pop.columns) == list(true_pop.columns)

#     ape_vals = []
#     used_combos = []
#     total_bins = 0
#     # Create contingency table for every combination of 2 variables
#     for col_1 in list(synthetic_pop.columns):
#         for col_2 in list(synthetic_pop.columns):

#             # Don't do contingency of the same column on itself or repeat tables
#             if col_1 == col_2:
#                 continue
#             elif [col_2,col_1] in used_combos or [col_1,col_2] in used_combos:
#                 continue
#             else:
#                 ct_synth = pd.crosstab(synthetic_pop[col_1], synthetic_pop[col_2], margins=False)
#                 ct_true = pd.crosstab(true_pop[col_1], true_pop[col_2], margins=False)
#                 used_combos.append([col_1, col_2])

#             # Calculate MPE on the contingency table
#             z = np.absolute((ct_true - ct_synth)/ct_true).values

#             # There may not be counts of certain bins in the synthetic population
#             nan_indices = np.argwhere(np.isnan(z))
#             nan_indices = [tuple(idx) for idx in nan_indices]
#             for idx in nan_indices:
#                 z[idx] = 1.0

#             ape_vals.append(np.sum(z))
#             total_bins += z.shape[0]*z.shape[1]

#     mape = np.sum(ape_vals) / total_bins
#     print(f"Bivariate (joint) MAPE: {mape}, Total Bins: {total_bins}")
#     return mape