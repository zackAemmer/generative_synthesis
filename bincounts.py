# Get value counts of a series according to the passed bins
# Used in place of sort_values to include counts of bins that are not present in dataset
# Bins are left-inclusive; right-most value is non-included upper limit
def get_bin_counts(series, bins):
    bin_counts = []
    for i in range(1,len(bins)):
        total = (((bins[i-1] <= series.values) & (series.values < bins[i])).sum())
        bin_counts.append(total)
    return bin_counts

# Lists of bins from common ACS tables
NP_bins = [1,2,3,4,99]
NP_bin_names = ['1','2','3','4+']
VEH_bins = [0,1,2,3,4,99]
VEH_bin_names = ['0','1','2','3','4+']
AGEP_bins = [15,20,25,45,55,60,99]
AGEP_bin_names = ['16-19','20-24','25-44','45-54','55-59','60+']
SEX_bins = [1,2,99]
SEX_bin_names = ['Male','Female']
SCHL_bins = [1,12,16,18,20,21,22,99]
SCHL_bin_names = ['1-11','12-15','16-17','18-19','20','21','22-24']
COW_bins = [1,3,6,8,9]
COW_bin_names = ['1-2','3-5','6-7','8']
RAC1P_bins = [1,2,3,6,7,8,9,99]
RAC1P_bin_names = ['1','2','3-5','6','7','8','9']
PINCP_bins = [-9999,10000,15000,25000,35000,50000,65000,75000,999999]
PINCP_bin_names = ['<10k','10k-14k','15k-24k','25k-34k','35k-49k','50k-64k','65k-74k','75k+']
bin_list = [NP_bins,VEH_bins,AGEP_bins,SEX_bins,SCHL_bins,COW_bins,RAC1P_bins,PINCP_bins]
bin_names_list = [NP_bin_names,VEH_bin_names,AGEP_bin_names,SEX_bin_names,SCHL_bin_names,COW_bin_names,RAC1P_bin_names,PINCP_bin_names]
var_list = ['NP','VEH','AGEP','SEX','SCHL','COW','RAC1P','PINCP']
var_names_list = ['Household Size','Household Vehicles','Age','Sex','Educational Attainment (>25yrs old)','Class of Worker','Race','Personal Income']