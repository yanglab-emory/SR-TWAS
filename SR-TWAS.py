#/usr/bin/env python

###############################################################
import argparse
import multiprocessing
import operator
import subprocess
import sys

from io import StringIO
from time import time

import numpy as np
import pandas as pd

# getting zetas
from scipy.optimize import minimize

# estimator classes
from sklearn.base import BaseEstimator
from sklearn.ensemble import StackingRegressor
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
parser.add_argument('--TIGAR_dir',type=str)

# eQTL weight files
parser.add_argument('--weights',dest='w_paths',nargs='+',default=[],type=str)
# ASSUME THESE FILES ARE *TABIXED*

# Test sampleID
parser.add_argument('--train_sampleID',type=str,dest='sampleid_path')

# specified chromosome number
parser.add_argument('--chr',type=str)

# # Gene annotation file
# parser.add_argument('--gene_anno',type=str,dest='annot_path')

# for Gene annotation and Expression level file
parser.add_argument('--gene_exp', type=str, dest='geneexp_path')

## test genofile
parser.add_argument('--genofile', type=str, dest='geno_path')

# specified input file type(vcf or dosages)
parser.add_argument('--genofile_type', type=str)

# 'DS' or 'GT' for VCF genotype file
parser.add_argument('--format', type=str)

# window
parser.add_argument('--window', type=int)

# missing rate: threshold for excluding SNPs with too many missing values
parser.add_argument('--missing_rate', type=float)

parser.add_argument('--cvR2', type=int)

# number of thread
parser.add_argument('--thread', type=int)

# Threshold of difference of maf between training data and testing data
parser.add_argument('--maf_diff', type=float)

# p-value for HW test
parser.add_argument('--hwe', type=float)

# output dir
parser.add_argument('--out_dir', type=str)

args = parser.parse_args()
sys.path.append(args.TIGAR_dir)

##########################
import TIGARutils as tg

# used to make entries for the dictionary used in reading in the weight files
def wk_dict_vals(w_paths, k, cols=None):
	# cols that will have weight number appended, rename dictionaries
	cols = ['ES','MAF'] if not cols else cols
	cols_k = [x+str(k) for x in cols]
	rename_cols = {x: x+str(k) for x in cols}

	# all columns in header, with renamed names
	wk_names = tg.get_header(w_paths[k], rename=rename_cols, zipped=w_paths[k].endswith('.gz'))
	wk_cols = ['CHROM','POS','REF','ALT','TargetID'] + cols_k

	if 'ID' in wk_names:
		wk_cols.append('ID')
	elif 'snpID' in wk_names:
		wk_cols.append('snpID')

	wk_dict = {
		'path': w_paths[k], 
		'names': wk_names,
		'cols': wk_cols}

	return wk_dict

def flatten(nested_list):
	return [j for i in nested_list for j in i]

# return R^2
def get_r2(y, predY, pval=False):
	lm = sm.OLS(y, sm.add_constant(predY)).fit()
	if pval:
		return lm.rsquared, lm.f_pvalue   
	return lm.rsquared

def _mse(y, ypred):
	mat = y - ypred
	return np.dot(mat, mat)

# formats list
#	[W0_N_SNP, W0_CVR2, W0_R2, W0_PVAL, ...,
#		..., WK_N_SNP, WK_CVR2, WK_R2, WK_PVAL]
# for output even when one or more of the W_ks lacks data for a target
def format_final_est_out_vals(target_k_outvals, target_ks, K):
	target_k_ind = {target_ks[j]: j for j in range(len(target_ks))}
	out_list = [target_k_outvals[target_k_ind[k]] if k in target_ks else (0, 0, 0, 1) for k in range(K)]
	return flatten(out_list)

def format_final_zetas(target_k_zetas, target_ks, K):
	target_k_ind = {target_ks[j]: j for j in range(len(target_ks))}
	out_list = [target_k_zetas[target_k_ind[k]] if k in target_ks else 0 for k in range(K)]
	return out_list

# estimator for individual training models
class WeightEstimator(BaseEstimator):
	_estimator_type = 'regressor'

	def __init__(self, raw_weights):
		self.raw_weights = raw_weights

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

	def est_out_vals(self, X, y):
		return [(self.n_features_in_, self.avg_r2_cv(X, y), *self.r2_pval(X, y))]

	def final_est_out_vals(self, X, y, ks, K):
		return format_final_est_out_vals((self.est_out_vals(X, y)), ks, K)

# final estimator for stacking regressor
class ZetasEstimator(BaseEstimator):
	_estimator_type = 'regressor'

	def __init__(self, min_method=None, tol=None):
		super().__init__()
		self.min_method = min_method
		self.tol = tol

	def _loss_func(self, zeta, X, y):
		if (len(zeta) == 1):
			Zeta = np.array([*zeta, 1 - np.sum(zeta)])
		else:
			Zeta = np.array(zeta)
		predY = np.dot(X, Zeta)
		R2 = get_r2(y, predY)
		return 1 - R2

	def fit(self, X, y, sample_weights=None):
		## FIX THIS TO DEAL WITH ALL 0 COLUMNS
		K = np.shape(X)[1]
		# if only one valid model set zeta = 1
		if (K == 1):
			self.coef_ = np.array([1])
			return self
		elif (K == 2):
			# initialize zeta list; all models weighted equally
			zeta_0 = np.full(K-1, 1/K)
			bnds = tuple([(0, 1)] * (K-1))
			# minimize loss function
			self.fit_res_ = minimize(
				self._loss_func,
				zeta_0,
				args=(X, y), 
				bounds=bnds,
				tol=self.tol,
				method=self.min_method)
			zeta = self.fit_res_.x
			self.coef_ = np.array([*zeta, 1 - np.sum(zeta)])
		else:
			zeta_0 = np.full(K, 1/K)
			bnds = tuple([(0, 1)] * K)
			cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})			
			# minimize loss function
			self.fit_res_ = minimize(
				self._loss_func,
				zeta_0,
				args=(X, y), 
				bounds=bnds,
				tol=self.tol,
				method=self.min_method,
				constraints = cons)
			zeta = self.fit_res_.x
			self.coef_ = np.array(zeta)
		return self

	def predict(self, X):
		return np.dot(X, self.coef_)

	def score(self, X, y):
		return get_r2(y, self.predict(X))

# stacking regressor
class WeightStackingRegressor(StackingRegressor):
	def __init__(self, estimators, final_estimator=ZetasEstimator(), *, cv=None,
		n_jobs=None, passthrough=False, verbose=0):
		super().__init__(
			estimators=estimators,
			final_estimator=final_estimator,
			cv=cv,
			n_jobs=n_jobs,
			passthrough=passthrough,
			verbose=verbose)

	def fit(self, X, y, sample_weights=None):
		self = super().fit(X, y, sample_weights)
		self.zetas_ = self.final_estimator_.coef_
		return self

	def score(self, X, y):
		return get_r2(y, self.predict(X))

	def r2_pval(self, X, y):
		return get_r2(y, self.predict(X), pval=True)

	def avg_r2_cv(self, X, y):
		return sum(cross_val_score(self, X, y)) / 5

	def est_avg_cv_scores(self, X, y):
		return [est[1].avg_r2_cv(X, y) for est in self.estimators]

	def est_r2_pvals(self, X, y):
		return list(zip(*[est[1].fit(X,y).r2_pval(X,y) for est in self.estimators]))

	def est_out_vals(self, X, y):
		est_n_snps = [est[1].fit().n_features_in_ for est in self.estimators] 
		est_avg_cv_score = self.est_avg_cv_scores(X, y)
		est_r2_pvals = self.est_r2_pvals(X, y)
		return list(zip(est_n_snps, est_avg_cv_score, *est_r2_pvals))

	def final_est_out_vals(self, X, y, ks, K):
		return format_final_est_out_vals(self.est_out_vals(X, y), ks, K)

#############################################################
# check input arguments
Kweights = len(args.w_paths)
if Kweights < 2:
	print('Must specify at least 2 weight files for stacked regression \n')

if args.genofile_type == 'vcf':
	gcol_sampleids_strt_ind = 9

	if (args.format != 'GT') and (args.format != 'DS'):
		raise SystemExit('Please specify the genotype data format used by the vcf file (--format ) as either "GT" or "DS".\n')
		
elif args.genofile_type == 'dosage':
	gcol_sampleids_strt_ind = 5
	args.format = 'DS'

else:
	raise SystemExit('Please specify the type input genotype file type (--genofile_type) as either "vcf" or "dosage".\n')

out_weight_path = args.out_dir + '/temp_CHR' + args.chr + '_stacked_eQTLweights.txt'
out_info_path = args.out_dir + '/CHR' + args.chr + '_StackedTrain_Info.txt'

#############################################################
# Print input arguments to log
print(
'''********************************
Input Arguments:
Gene Annotation and Expression file: {geneexp_path}
Training sampleID file: {sampleid_path}
Chromosome: {chr}
K (number of trained input models): {K}
cis-eQTL weight files:{w_paths_str}
Training genotype file: {geno_path}
Genotype file used for training is type: {genofile_type}
Genotype data format: {format}
Gene training region SNP inclusion window: +-{window}
Excluding SNPs if missing rate exceeds: {missing_rate}
Excluding SNPs matched between eQTL weight file and training genotype file if MAF difference exceeds: |{maf_diff}|
HWE p-value threshold for SNP inclusion: {hwe}
{cvR2_str} Stacked Regression model by 5-fold cross validation.
Number of threads: {thread}
Output directory: {out_dir}
Output training info file: {out_info}
Output trained weights file: {out_weight}
********************************'''.format(
	**args.__dict__,
	cvR2_str = {0:'Skipping evaluation of', 1:'Evaluating'}[args.cvR2],
	w_paths_str = '\n  '.join(args.w_paths),
	K = Kweights,
	out_info = out_info_path,
	out_weight = out_weight_path))

##########################
# STUFF FOR TEST FILES
##########################
# Gene Expression header, sampleIDs
print('Reading genotype, expression file headers.\n')
exp_cols = tg.get_header(args.geneexp_path)
exp_info_cols = list(exp_cols[:5])
exp_sampleids = exp_cols[5:]

# genofile header, sampleIDs
try:
	g_cols = tg.call_tabix_header(args.geno_path)
except: 
	g_cols = tg.get_header(args.geno_path, zipped=True)
gcol_sampleids = g_cols[gcol_sampleids_strt_ind:]

# geno, exp overlapping sampleIDs
gcol_exp_sampleids = np.intersect1d(gcol_sampleids, exp_sampleids)

if not gcol_exp_sampleids.size:
	raise SystemExit('The gene expression file and genotype file have no sampleIDs in common.')

# use the overlapping sampleIDs in the genotype and expression file
print('Reading sampleID file.\n')
spec_sampleids = pd.read_csv(
	args.sampleid_path,
	sep='\t',
	header=None)[0].drop_duplicates()

print('Matching sampleIDs.\n')
sampleID = np.intersect1d(spec_sampleids, gcol_exp_sampleids)

if not sampleID.size:
	raise SystemExit('There are no overlapped sample IDs between the gene expression file, genotype file, and sampleID file.')

# get columns to read in
g_cols_ind, g_dtype = tg.genofile_cols_dtype(g_cols, args.genofile_type, sampleID)
e_cols, e_dtype = tg.exp_cols_dtype(exp_cols, sampleID)

### EXPRESSION FILE
# ### Extract expression level by chromosome
print('Reading gene expression data.\n')
try:
	GeneExp_chunks = pd.read_csv(
		args.geneexp_path, 
		sep='\t',
		header=0,
		iterator=True, 
		chunksize=10000,
		usecols=e_cols,
		dtype=e_dtype)

	GeneExp = pd.concat([x[x['CHROM']==args.chr] for x in GeneExp_chunks],
		ignore_index=True)

except:
	GeneExp_chunks = pd.read_csv(
		args.geneexp_path, 
		sep='\t',
		header=None,
		iterator=True,
		skiprows=0,
		chunksize=10000,
		usecols=e_cols)

	GeneExp = pd.concat([x[x[0]==args.chr] for x in GeneExp_chunks],
		ignore_index=True).astype(e_dtype)

	GeneExp.columns = [exp_cols[i] for i in GeneExp.columns]

if GeneExp.empty:
	raise SystemExit('There are no valid gene expression training data for chromosome ' + args.chr + '\n')

GeneExp = tg.optimize_cols(GeneExp)

TargetID = GeneExp.TargetID.values
n_targets = TargetID.size

##########################
wdict = {k: {**wk_dict_vals(args.w_paths, k)} for k in range(Kweights)}

weight_dtypes = {
	'CHROM': object,
	'POS': np.int64,
	'REF': object,
	'ALT': object,
	'TargetID': object,
	'ID': object,
	'snpID':object,
	**{x+str(k): np.float64 for x in ['ES','MAF'] for k in range(Kweights)}  
	}

ES_cols = ['ES'+str(k) for k in range(Kweights)]
MAF_cols = ['MAF'+str(k) for k in range(Kweights)]

##########################
print('Creating file: ' + out_weight_path + '\n')
out_weight_cols = ['CHROM','POS','ID','REF','ALT','TargetID','MAF','p_HWE','ES']
pd.DataFrame(columns=out_weight_cols).to_csv(
	out_weight_path,
	header=True,
	index=None,
	sep='\t',
	mode='w')

# info output file
out_info_dtypes_dict = {
	'_N_SNP': np.int64, '_CVR2': np.float64,
	'_R2': np.float64, '_PVAL': np.float64}

zeta_out_cols = ['Z'+str(k) for k in range(Kweights)]
wk_out_info_dtypes = {'W'+str(k)+col: out_info_dtypes_dict[col] for k in range(Kweights) for col in out_info_dtypes_dict}

wk_out_info_cols = list(wk_out_info_dtypes.keys())

print('Creating file: ' + out_info_path + '\n')
out_info_cols = ['CHROM', 'GeneStart', 'GeneEnd', 'TargetID', 'GeneName','sample_size','N_SNP','N_EFFECT_SNP','CVR2', 'R2', 'PVAL'] + zeta_out_cols + wk_out_info_cols

pd.DataFrame(columns=out_info_cols).to_csv(
	out_info_path,
	sep='\t',
	header=True,
	index=None,
	mode='w')

print('********************************\n')

##############################################################
# thread function

@tg.error_handler
def thread_process(num):
	
	target = TargetID[num]
	print('num=' + str(num) + '\nTargetID=' + target)
	target_exp = GeneExp.iloc[[num]]
	# center data
	target_exp = tg.center(target_exp, sampleID)

	start = str(max(int(target_exp.GeneStart) - args.window, 0))
	end = str(int(target_exp.GeneEnd) + args.window)

	g_proc_out = tg.call_tabix(args.geno_path, args.chr, start, end)

	if not g_proc_out:
		print('There is no genotype data for TargetID: ' + target + '\n')
		return None

	target_geno = pd.read_csv(
		StringIO(g_proc_out.decode('utf-8')),
		sep='\t',
		low_memory=False,
		header=None,
		usecols=g_cols_ind,
		dtype=g_dtype,
		na_values=['.', '.|.', './.'])

	target_geno.columns = [g_cols[i] for i in target_geno.columns]
	target_geno = tg.optimize_cols(target_geno)

	# get snpIDs
	target_geno['snpID'] = tg.get_snpIDs(target_geno)
	target_geno = target_geno.drop_duplicates(['snpID'], keep='first', ignore_index=True).reset_index(drop=True)

	# read in weight data for target from all K weight files
	weights = pd.DataFrame()
	target_ks = []
	empty_target_ks = []

	for wk in wdict:
		compress_type = 'gzip' if wdict[wk]['path'].endswith('.gz') else None

		weightk_chunks = pd.read_csv(
			wdict[wk]['path'], 
			sep='\t', 
			iterator=True, 
			chunksize=10000,
			header=0,
			compression = compress_type,
			names=wdict[wk]['names'],
			usecols=wdict[wk]['cols'],
			dtype=weight_dtypes)

		weightk = pd.concat([x[x['TargetID'].apply(lambda x:x.startswith(target))] for x in weightk_chunks], ignore_index=True)

		# remove gene versions
		weightk['TargetID'] = weightk.TargetID.apply(lambda x:x.split('.')[0])

		weightk = tg.optimize_cols(weightk)

		generated_snpID = False

		if 'ID' in weightk.columns:
			weightk = weightk.rename(columns={'ID':'snpID'})

		if not 'snpID' in weightk.columns:
			weightk['snpID'] = tg.get_snpIDs(weightk)
			generated_snpID = True

		snp_overlap = np.intersect1d(weightk.snpID, target_geno.snpID)

		if not snp_overlap.size and not generated_snpID:
			# may be bad ID/snpID column in file
			weightk['snpID'] = tg.get_snpIDs(weightk)
			snp_overlap = np.intersect1d(weightk.snpID, target_geno.snpID)

		weightk = weightk[weightk.snpID.isin(snp_overlap)].drop_duplicates(['snpID'],keep='first').reset_index(drop=True)

		weightk = weightk.drop(columns=['CHROM','POS','REF','ALT','TargetID'])

		if weightk.empty:
			empty_target_ks.append(wk)

		else:
			target_ks.append(wk)

		if weights.empty:
			weights = weightk

		else:
			# weights = weights.merge(weightk, how='outer',on=['CHROM','POS','REF','ALT','TargetID','snpID'])
			weights = weights.merge(weightk, how='outer', on=['snpID'])

	if weights.empty:
		print('No cis-eQTL weights for TargetID: ' + target + '\n')
		return None

	# weights = weights.drop(columns=['CHROM','POS','REF','ALT','TargetID'])

	empty_wk_cols = flatten([['ES'+str(k),'MAF'+str(k)] for k in empty_target_ks])
	weights[empty_wk_cols] = np.nan

	### 
	# get overlapping snps, filter both files
	snp_overlap = np.intersect1d(weights.snpID, target_geno.snpID)

	if not snp_overlap.size:
		print('No overlapping test SNPs between weight files and genotype file for TargetID: ' + target + '\n')
		return None

	weights = weights[weights.snpID.isin(snp_overlap)].reset_index(drop=True)
	target_geno = target_geno[target_geno.snpID.isin(snp_overlap)].reset_index(drop=True)
	### 

	# finish processing genofile
	if args.genofile_type == 'vcf':
		target_geno = tg.check_prep_vcf(target_geno, args.format, sampleID)

	# reformat sample values
	target_geno = tg.reformat_sample_vals(target_geno, args.format, sampleID)

	# filter out variants that exceed missing rate threshold
	target_geno = tg.handle_missing(target_geno, sampleID, args.missing_rate)

	# maf
	target_geno = tg.calc_maf(target_geno, sampleID, 0, op=operator.ge)

	# get, filter p_HWE
	target_geno = tg.calc_p_hwe(target_geno, sampleID, args.hwe)

	# center data
	target_geno = tg.center(target_geno, sampleID)

	### 
	# get overlapping snps after potential filtering, filter both files
	# snp_overlap = np.intersect1d(weights.snpID, target_geno.snpID)

	# if not snp_overlap.size:
	#     print('No overlapping test SNPs between weight files and genotype file after filtering for TargetID: ' + target + '\n')
	#     return None

	# weights = weights[weights.snpID.isin(snp_overlap)].reset_index(drop=True)
	# target_geno = target_geno[target_geno.snpID.isin(snp_overlap)].reset_index(drop=True)
	# ### 

	# merge datasets (inner in case missing snpIDs after filtering)
	train = target_geno.merge(weights,
		left_on='snpID', 
		right_on='snpID', 
		how='inner').set_index('snpID')

	# filter by MAF diff - if an MAF# differs too much, set ES# to nan
	for k in range(Kweights):
		exceeds_maf_diff = np.abs(train[MAF_cols[k]].values-train['MAF'].values) > args.maf_diff
		train.loc[exceeds_maf_diff, ES_cols[k]] = np.nan

	# filter out snps where all weights are nan
	all_missing_weights = train[ES_cols].count(axis=1) == 0
	train = train[~all_missing_weights]
	n_snps = train.index.size

	if not n_snps:
		print('All SNP MAFs for training data and testing data differ by a magnitude greater than ' + str(args.maf_diff) + ' for TargetID: ' + target + '\n')
		return None

	# do stacked regression
	print('Performing Stacked Regression.')
	# target_sampleid = np.intersect1d(train.columns.isin(sampleID), target_exp.columns.isin(sampleID))
	# X = train[target_sampleid].T
	# Y = target_exp[target_sampleid].values.ravel()
	X = train[sampleID].T
	Y = target_exp[sampleID].values.ravel()

	if len(target_ks) == 1:
		# only one weight model with usable data, don't need to do stacking
		# 'Only trained model ' + str(int(target_ks[0])) + ' has usable weights for TargetID: ' + target + '\n'
		print('Only trained model ' + str(target_ks[0]) + ' has usable weights for TargetID: ' + target + '\n')
		reg = WeightEstimator(train[ES_cols[target_ks[0]]]).fit(X, Y)
		reg.zetas_ = [1]

	else:
		# do stacking
		weight_estimators = [(str(k), WeightEstimator(train[ES_cols[k]])) for k in target_ks]
		reg = WeightStackingRegressor(estimators=weight_estimators).fit(X, Y)

	# 5-FOLD CROSS-VALIDATION
	if args.cvR2:
		print('Running 5-fold CV.')
		avg_r2_cv = reg.avg_r2_cv(X, Y)

		print('Average R2 for 5-fold CV: {:.4f}'.format(avg_r2_cv))
		if avg_r2_cv < 0.005:
			print('Average R2 < 0.005; Skipping training for TargetID: ' + target + '\n')
			return None
	else:
		avg_r2_cv = 0
		print('Skipping evaluation by 5-fold CV average R2...')

	stacked_r2, stacked_pval = reg.r2_pval(X, Y)
	zetas = format_final_zetas(reg.zetas_, target_ks, Kweights)
	wk_out_vals = reg.final_est_out_vals(X, Y, ks=target_ks, K=Kweights)

	# output trained weights
	out_weights = train[['CHROM','POS','REF','ALT','MAF','p_HWE']].copy()
	out_weights['ID'] = out_weights.index
	out_weights['TargetID'] = target
	out_weights['ES'] = np.dot(train[ES_cols].fillna(0), zetas)

	# filter for non-zero effect size
	out_weights = out_weights[out_weights['ES']!=0]

	# reorder columns
	out_weights = out_weights[out_weight_cols]

	out_weights.to_csv(
		out_weight_path,
		sep='\t',
		header=None,
		index=None,
		mode='a')

	# output training info
	train_info = target_exp[exp_info_cols].copy()
	train_info['sample_size'] = Y.size
	train_info['ST_N_SNP'] = n_snps
	train_info['ST_N_EFFECT_SNP'] = out_weights.ES.size
	train_info['ST_CVR2'] = avg_r2_cv
	train_info['ST_R2'] = stacked_r2
	train_info['ST_PVAL'] = stacked_pval

	train_info[zeta_out_cols] = zetas
	train_info[wk_out_info_cols] = wk_out_vals

	train_info = train_info.astype(wk_out_info_dtypes)

	train_info.to_csv(
		out_info_path,
		sep='\t',
		header=None,
		index=None,
		mode='a')

	print('Target training completed.\n')


###############################################################
# thread process
if __name__ == '__main__':
	print('Starting training for ' + str(n_targets) + ' target genes.\n')
	pool = multiprocessing.Pool(args.thread)
	pool.imap(thread_process,[num for num in range(n_targets)])
	pool.close()
	pool.join()
	print('Done.')


###############################################################
# time calculation
elapsed_sec = time()-start_time
elapsed_time = tg.format_elapsed_time(elapsed_sec)
print('Computation time (DD:HH:MM:SS): ' + elapsed_time)

