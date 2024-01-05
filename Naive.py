#/usr/bin/env python

###############################################################
import argparse
import multiprocessing
import operator
import sys

from time import time

import numpy as np
import pandas as pd

# getting zetas
from scipy.optimize import minimize

# estimator classes
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

# R2, Pval
import statsmodels.api as sm

###############################################################
# time calculation
start_time = time()

###############################################################
# parse input arguments
parser = argparse.ArgumentParser(description='SR-TWAS script.')

# Specify tool directory
parser.add_argument('--SR_TWAS_dir', type=str)

# eQTL weight files
parser.add_argument('--weights', dest='w_paths', nargs='+', default=[], type=str)
# ASSUME THESE FILES ARE *TABIXED*

# weight names - names for base models
parser.add_argument('--weights_names', dest='w_names', nargs='+', default=[], type=str)

# Test sampleID
parser.add_argument('--train_sampleID', type=str, dest='sampleid_path')

# specified chromosome number
parser.add_argument('--chr', type=str, dest='chrm')

# for Gene annotation and Expression level file
parser.add_argument('--gene_exp', type=str, dest='geneexp_path')

## test genofile
parser.add_argument('--genofile', type=str, dest='geno_path')

# specified input file type(vcf or dosages)
parser.add_argument('--genofile_type', type=str)

# 'DS' or 'GT' for VCF genotype file
parser.add_argument('--format', type=str, dest='data_format')

# window
parser.add_argument('--weight_threshold', type=float)

# window
parser.add_argument('--window', type=int)

# missing rate: threshold for excluding SNPs with too many missing values
parser.add_argument('--missing_rate', type=float)

# cvR2 (0: no CV, 1: CV)
parser.add_argument('--cvR2', type=int)

# threshold cvR2 value for training (default: 0.005)
parser.add_argument('--cvR2_threshold', type=float)

# number of parallel processes
parser.add_argument('--parallel', type=int)

# Threshold of difference of maf between training data and testing data
parser.add_argument('--maf_diff', type=float)

# p-value for HW test
parser.add_argument('--hwe', type=float)

# output dir
parser.add_argument('--out_dir', type=str)

# file names
parser.add_argument('--out_weight_file', type=str)
parser.add_argument('--out_info_file', type=str)

args = parser.parse_args()
sys.path.append(args.SR_TWAS_dir)

##########################
import TIGARutils as tg

def W(k):
	return 'W'+str(k)

def ES(k):
	return 'ES'+str(k)

def MAF(k):
	return 'MAF'+str(k)

def flatten(nested_list):
	return [j for i in nested_list for j in i]

def merge_dicts(dict_list):
	return_dict = {}
	for d in dict_list:
		return_dict.update(d)
	return return_dict

# return R^2
def get_r2(y, predY, pval=False):
	lm = sm.OLS(y, sm.add_constant(predY)).fit()
	if pval:
		return lm.rsquared, lm.f_pvalue   
	return lm.rsquared

def _mse(y, ypred):
	mat = y - ypred
	return np.dot(mat, mat)

def final_wk_out_vals(target_k_outvals, K):
	out_dict = merge_dicts([{W(k)+'_N_SNP': 0, W(k)+'_CVR2': 0, W(k)+'_R2': 0, W(k)+'_PVAL': 1} for k in range(K)])
	out_dict.update(target_k_outvals)
	return out_dict

# estimator for individual training models
class WeightEstimator(BaseEstimator):
	_estimator_type = 'regressor'

	def __init__(self, raw_weights, name):
		self.raw_weights = raw_weights
		self.name = name

	def fit(self, X=None, y=None):
		self.coef_ = self.raw_weights.dropna()
		self.snpID = self.coef_.index.values
		self.n_features_in_  = self.coef_.size
		return self

	def predict(self, X):
		return np.dot(X[self.snpID], self.coef_)

	def score(self, X, y):
		return get_r2(y, self.predict(X))

	def r2_pval(self, X, y):
		return get_r2(y, self.predict(X), pval=True)

	def avg_r2_cv(self, X, y):
		return sum(cross_val_score(self, X, y)) / 5

	def out_vals(self, X, y):
		r2, pval = self.r2_pval(X, y)
		return {self.name+'_N_SNP': self.n_features_in_, 
			self.name+'_CVR2': self.avg_r2_cv(X, y), 
			self.name+'_R2': r2, 
			self.name+'_PVAL': pval}


#############################################################
# check input arguments
Kweights = len(args.w_paths)
if Kweights < 2:
	raise SystemExit('Must specify at least 2 weight files for stacked regression.\n')


# check for incorrect number of weight names or non-unique names
if (len(args.w_names) != Kweights) or (len(args.w_names) != len(set(args.w_names))):

	old_w_names = args.w_names
	args.w_names = ['W'+str(k) for k in range(Kweights)]

	# only show warning if user supplied --weights_names (ie, args.w_names was non-empty)
	if old_w_names:
		w_names_warning = ''

		if (len(old_w_names) != Kweights):
			w_names_warning = str(len(old_w_names)) + ' names given for ' + str(Kweights) + ' weight files; '

		if (len(old_w_names) != len(set(old_w_names))):
			w_names_warning = w_names_warning + 'non-unique names; '
		
		print('WARNING: Invalid --weight_names (' + ', '.join(old_w_names) + '); ' + w_names_warning + 'reverting to default names (' + ', '.join(args.w_names) + ').\n')


if args.genofile_type == 'vcf':
	if (args.data_format != 'GT') and (args.data_format != 'DS'):
		raise SystemExit('Please specify the genotype data format used by the vcf file (--format ) as either "GT" or "DS".\n')
		
elif args.genofile_type == 'dosage':
	args.data_format = 'DS'

else:
	raise SystemExit('Please specify the type input genotype file type (--genofile_type) as either "vcf" or "dosage".\n')

out_weight_path = args.out_dir + '/temp_' + args.out_weight_file
out_info_path = args.out_dir + '/' + args.out_info_file

do_maf_diff = 0 if not args.maf_diff else 1

#############################################################
# Print input arguments to log
print(
'''********************************
Input Arguments:
Gene Annotation and Expression file: {geneexp_path}
Training sampleID file: {sampleid_path}
Chromosome: {chrm}
K (number of trained input models): {K}
cis-eQTL weight files:{w_paths_str}
cis-eQTL model names: {w_names_str}
Training genotype file: {geno_path}
Genotype file used for training is type: {genofile_type}
Genotype data format: {data_format}
Gene training region SNP inclusion window: +-{window}
Excluding SNPs if missing rate exceeds: {missing_rate}
{maf_diff_str1}xcluding SNPs matched between eQTL weight file and training genotype file {maf_diff_str2}
HWE p-value threshold for SNP inclusion: {hwe}
{cvR2_str1} Stacked Regression model by 5-fold cross validation{cvR2_str2}.
Number of parallel processes: {parallel}
Output directory: {out_dir}
Output training info file: {out_info}
Output trained weights file: {out_weight}
********************************'''.format(
	**args.__dict__,
	cvR2_str1 = {0:'Skipping evaluation of', 1:'Evaluating'}[args.cvR2],
	cvR2_str2 = {0:'', 1:' with inclusion threshold Avg.CVR2 >' + str(args.cvR2_threshold)}[args.cvR2],
	maf_diff_str1 = {0:'Not e', 1:'E'}[do_maf_diff],
	maf_diff_str2 = {0:'by MAF difference.', 1:'if MAF difference exceeds: |' + str(args.maf_diff) + '|'}[do_maf_diff],
	w_paths_str = '\n  '.join(args.w_paths),
	w_names_str = ', '.join(args.w_names),
	K = Kweights,
	out_info = out_info_path,
	out_weight = out_weight_path))

##########################
# STUFF FOR TEST FILES
##########################
# Gene Expression header, sampleIDs
print('Reading genotype, expression file headers, sample IDs.\n')
sampleID, sample_size, geno_info, exp_info = tg.sampleid_startup(**args.__dict__)

# Read in gene expression info
print('Reading gene expression data.\n')
GeneExp, TargetID, n_targets = tg.read_gene_annot_exp(**exp_info)

# get info for the weight files
weights_info = tg.weight_k_files_info(**args.__dict__)

ES_cols = [ES(k) for k in range(Kweights)]

# set up output files
print('Creating file: ' + out_weight_path + '\n')
out_weight_cols = ['CHROM','POS','ID','REF','ALT','TargetID','MAF','p_HWE','ES']
pd.DataFrame(columns=out_weight_cols).to_csv(
	out_weight_path,
	header=True,
	index=None,
	sep='\t',
	mode='w')

# dictionary of dtypes for info output
info_wk_dtypes = merge_dicts([{W(k)+'_N_SNP':np.int64, W(k)+'_CVR2':np.float64, W(k)+'_R2':np.float64, W(k)+'_PVAL':np.float64} for k in range(Kweights)])
info_wk_cols = list(info_wk_dtypes.keys())
info_cols = ['CHROM','GeneStart','GeneEnd','TargetID','GeneName','sample_size','N_SNP','N_EFFECT_SNP','CVR2','R2','PVAL'] + info_wk_cols

out_info_wk_cols = [name_k+v for name_k in args.w_names for v in ['_NSNP','_CVR2','_R2','_PVAL']]
out_info_cols = ['CHROM','GeneStart','GeneEnd','TargetID','GeneName','sample_size','N_SNP','N_EFFECT_SNP','CVR2','R2','PVAL'] + out_info_wk_cols

print('Creating file: ' + out_info_path + '\n')
pd.DataFrame(columns=out_info_cols).to_csv(
	out_info_path,
	sep='\t',
	header=True,
	index=None,
	mode='w')

print('********************************\n')

##############################################################
# parallel function

@tg.error_handler
def parallel_process(num):
	
	target = TargetID[num]
	print('num=' + str(num) + '\nTargetID=' + target)
	Expr = GeneExp.iloc[[num]]

	# center data
	Expr = tg.center(Expr, sampleID)

	start = str(max(int(Expr.GeneStart) - args.window, 0))
	end = str(int(Expr.GeneEnd) + args.window)

	# Query files; function stops here if no data in genotype and/or no weight file data
	tabix_target_ks, empty_target_ks = tg.tabix_query_files(start, end, **args.__dict__)

	# READ IN AND PROCESS GENOTYPE DATA 
	# file must be bgzipped and tabix
	Geno = tg.read_tabix(start, end, sampleID, **geno_info)

	# read in weight data for target from all K weight files
	Weights = pd.DataFrame()
	target_ks = []

	for k in tabix_target_ks:
		# read in from tabix, may be empty
		W_k = tg.read_tabix(start, end, target=target, raise_error=False, **weights_info[k])

		while not W_k.empty:
			## CURRENTLY IGNORES ENSEMBLE VERSION
			# ## CONSIDER CHECKING IF ('.' in target) and requiring ensemble version if specified in the gene annotation file; otherwise use whatever ensemble version in the weight file
			# if np.any(W_k.TargetID.str.contains('.')):
			# 	W_k['TargetID'] = W_k.TargetID.str.split('.', expand=True)[0]

			W_k['snpIDflip'] = tg.get_snpIDs(W_k, flip=True)

			W_k = W_k[np.any(W_k[['snpID','snpIDflip']].isin(Geno.snpID.values), axis=1)].reset_index(drop=True)

			# if not in Geno.snpIDs, assumed flipped; if flipped, 1 - MAF and -ES
			flip = np.where(W_k.snpID.isin(Geno.snpID), True, False)

			if not np.all(flip):
				# set correct snpID, MAF, ES
				W_k['snpID'] = np.where(flip, W_k.snpID, W_k.snpIDflip)
				if args.maf_diff:
					W_k[MAF(k)] = np.where(flip, W_k[MAF(k)], 1 - W_k[MAF(k)])
				W_k[ES(k)] = np.where(flip, W_k[ES(k)], -W_k[ES(k)])

			W_k = W_k.drop(columns=['CHROM','POS','REF','ALT','TargetID','snpIDflip'])

			target_ks.append(k)

			if Weights.empty:
				Weights = W_k
			else:
				Weights = Weights.merge(W_k, how='outer', on=['snpID'])

			break

		else:
			empty_target_ks.append(k)

	if Weights.empty:
		print('No cis-eQTL weights with snp overlap in genotype data for target.\n')
		return None

	# ks may be added in read in step, need to sort for correct output later
	empty_target_ks.sort()

	# add nan columns for weight files without data
	if args.maf_diff:
		empty_wk_cols = flatten([[ES(k),MAF(k)] for k in empty_target_ks])
	else: 
		empty_wk_cols = [ES(k) for k in empty_target_ks]
	Weights[empty_wk_cols] = np.nan

	# get overlapping snps, filter Geno
	Geno = Geno[Geno.snpID.isin(Weights.snpID.values)].reset_index(drop=True)

	# filter out variants that exceed missing rate threshold
	Geno = tg.handle_missing(Geno, sampleID, args.missing_rate)

	# maf
	Geno = tg.calc_maf(Geno, sampleID, 0, op=operator.ge)

	# get, filter p_HWE
	Geno = tg.calc_p_hwe(Geno, sampleID, args.hwe)

	# center data
	Geno = tg.center(Geno, sampleID)

	# merge datasets (inner in case missing snpIDs after filtering)
	Train = Geno.merge(Weights,
		left_on='snpID', 
		right_on='snpID', 
		how='inner').set_index('snpID')

	# filter by MAF diff - if an MAF# differs too much, set ESk to nan
	if args.maf_diff:
		for k in range(Kweights):
			maf_diff = np.abs(Train[MAF(k)].values - Train['MAF'].values)
			Train.loc[maf_diff > args.maf_diff, ES(k)] = np.nan

	# filter out snps where all weights are nan
	all_missing_weights = Train[ES_cols].count(axis=1) == 0
	Train = Train[~all_missing_weights]
	n_snps = Train.index.size

	if not n_snps:
		print('No valid SNPs{}.\n'.format({0:'', 1:'; All SNP MAFs for training data and testing data differ by a magnitude greater than ' + str(args.maf_diff)}[do_maf_diff]))
		return None

	# do stacked regression
	print('Performing naive stacking.')
	X = Train[sampleID].T
	Y = Expr[sampleID].values.ravel()

	# get base model info
	if len(target_ks) == 1:
		print('Only trained model ' + str(target_ks[0]) + ' has usable weights for target.')
		reg_temp = WeightEstimator(Train[ES(target_ks[0])], W(target_ks[0])).fit(X, Y)
		wk_out_vals = final_wk_out_vals(reg_temp.out_vals(X, Y), Kweights)
	else:
		wk_out_vals = final_wk_out_vals(merge_dicts([WeightEstimator(Train[ES(k)], W(k)).fit().out_vals(X, Y) for k in target_ks]), Kweights)

	# if len(target_ks) == 1:
	# 	# only one weight model with usable data, don't need to do stacking
	# 	print('Only trained model ' + str(target_ks[0]) + ' has usable weights for target.')
	# 	Train['Naive'] = Train[ES(target_ks[0])]
	# 	reg = WeightEstimator(Train[ES(target_ks[0])], W(target_ks[0])).fit(X, Y)
	# 	wk_out_vals = final_wk_out_vals(reg.out_vals(X, Y), Kweights)

	# else:
	# 	# Train['Naive'] = np.nanmean(Train[ES_cols].values, axis=1)
	# 	## impute missing as 0
	# 	Train['Naive'] = np.mean(np.nan_to_num(Train[ES_cols].values), axis=1)

	# 	reg = WeightEstimator(Train['Naive'], 'Naive').fit(X, Y)

	# 	wk_out_vals = final_wk_out_vals(merge_dicts([WeightEstimator(Train[ES(k)], W(k)).fit().out_vals(X, Y) for k in target_ks]), Kweights)

	Train['Naive'] = np.mean(np.nan_to_num(Train[ES_cols].values), axis=1)
	reg = WeightEstimator(Train['Naive'], 'Naive').fit(X, Y)
	

	# 5-FOLD CROSS-VALIDATION
	if args.cvR2:
		print('Running 5-fold CV.')
		avg_r2_cv = reg.avg_r2_cv(X, Y)

		print('Average R2 for 5-fold CV: {:.4f}'.format(avg_r2_cv))
		if avg_r2_cv < args.cvR2_threshold:
			print('Average R2 < ' + str(args.cvR2_threshold) + '; Skipping training for target.\n')
			return None
	else:
		avg_r2_cv = 0
		print('Skipping evaluation by 5-fold CV average R2...')

	naive_r2, naive_pval = reg.r2_pval(X, Y)

	# output Trained weights
	Weight_Out = Train[['CHROM','POS','REF','ALT','MAF','p_HWE']].copy()
	Weight_Out['ID'] = Weight_Out.index
	Weight_Out['TargetID'] = target
	Weight_Out['ES'] = Train['Naive']
	# filter for non-zero effect size
	Weight_Out = Weight_Out[Weight_Out['ES'] != 0]
	# reorder columns
	Weight_Out = Weight_Out[out_weight_cols]
	Weight_Out.to_csv(
		out_weight_path,
		sep='\t',
		header=None,
		index=None,
		mode='a')

	# output training info
	Info = Expr[['CHROM','GeneStart','GeneEnd','TargetID','GeneName']].copy().reset_index()
	Info['sample_size'] = Y.size
	Info['N_SNP'] = n_snps
	Info['N_EFFECT_SNP'] = Weight_Out.ES.size
	Info['CVR2'] = avg_r2_cv
	Info['R2'] = naive_r2
	Info['PVAL'] = naive_pval
	
	Info = Info.join(pd.DataFrame.from_records(wk_out_vals, index=[0]).astype(info_wk_dtypes))[info_cols]

	Info.to_csv(
		out_info_path,
		sep='\t',
		header=None,
		index=None,
		mode='a')

	print('Target training completed.\n')

###############################################################
# parallel process
if __name__ == '__main__':
	print('Starting training for ' + str(n_targets) + ' target genes.\n')
	pool = multiprocessing.Pool(args.parallel)
	pool.imap(parallel_process,[num for num in range(n_targets)])
	pool.close()
	pool.join()
	print('Done.')


###############################################################
# time calculation
elapsed_sec = time()-start_time
elapsed_time = tg.format_elapsed_time(elapsed_sec)
print('Computation time (DD:HH:MM:SS): ' + elapsed_time)

