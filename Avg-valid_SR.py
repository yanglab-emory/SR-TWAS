#/usr/bin/env python

###############################################################
import argparse
import multiprocessing
import operator
import sys

from time import time

import numpy as np
import pandas as pd

###############################################################
# time calculation
start_time = time()

## SINCE NO VALIDATION DATA USES WEIGHT MODEL 0 AS THE REFERENCE SNP MODEL

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

# specified chromosome number
parser.add_argument('--chr', type=str, dest='chrm')

# # Gene annotation file
parser.add_argument('--gene_anno', type=str, dest='annot_path', required=True)

# window
parser.add_argument('--window', type=int)

# number of thread
parser.add_argument('--thread', type=int)

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

def merge_dicts(dict_list):
	return_dict = {}
	for d in dict_list:
		return_dict.update(d)
	return return_dict

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

out_weight_path = args.out_dir + '/temp_' + args.out_weight_file
out_info_path = args.out_dir + '/' + args.out_info_file

#############################################################
# Print input arguments to log
print(
'''********************************
Input Arguments:
Gene Annotation file: {annot_path}
Chromosome: {chrm}
K (number of trained input models): {K}
cis-eQTL weight files:{w_paths_str}
cis-eQTL model names: {w_names_str}
Number of threads: {thread}
Output directory: {out_dir}
Output training info file: {out_info}
Output trained weights file: {out_weight}
********************************'''.format(
	**args.__dict__,
	w_paths_str = '\n  '.join(args.w_paths),
	w_names_str = ', '.join(args.w_names),
	K = Kweights,
	out_info = out_info_path,
	out_weight = out_weight_path))

##########################
# STUFF FOR TEST FILES
##########################

# Read in gene annotation info
print('Reading gene annotation data.\n')
Gene, TargetID, n_targets = tg.read_gene_annot_exp(**args.__dict__)

# get info for the weight files
weights_info = tg.weight_k_files_info(**args.__dict__)

ES_cols = [ES(k) for k in range(Kweights)]

# set up output files
print('Creating file: ' + out_weight_path + '\n')

out_weight_dtypes = {'CHROM':np.int8, 'POS':np.int64, 'ID':object, 'REF':object ,'ALT':object, 'TargetID':object, 'ES':np.float64}
out_weight_cols = ['CHROM','POS','ID','REF','ALT','TargetID','ES']

pd.DataFrame(columns=out_weight_cols).to_csv(
	out_weight_path,
	header=True,
	index=None,
	sep='\t',
	mode='w')

# dictionary of dtypes for info output
info_wk_dtypes = merge_dicts([{W(k)+'_N_SNP':np.int64} for k in range(Kweights)])

info_wk_cols = list(info_wk_dtypes.keys())
info_cols = ['CHROM','GeneStart','GeneEnd','TargetID','GeneName','N_SNP','N_EFFECT_SNP'] + info_wk_cols


out_info_wk_cols = [name_k+'_NSNP'for name_k in args.w_names]
out_info_cols = ['CHROM','GeneStart','GeneEnd','TargetID','GeneName','N_SNP','N_EFFECT_SNP'] + out_info_wk_cols

print('Creating file: ' + out_info_path + '\n')
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
	Gene_Info = Gene.iloc[[num]]

	start = str(max(int(Gene_Info.GeneStart) - args.window, 0))
	end = str(int(Gene_Info.GeneEnd) + args.window)

	# Query files; function stops here if no data in genotype and/or no weight file data
	tabix_target_ks, empty_target_ks = tg.tabix_query_files(start, end, **args.__dict__)

	# read in weight data for target from all K weight files
	Weights = pd.DataFrame()
	target_ks = []
	wk_n_snps = {}

	for k in tabix_target_ks:
		new_snpID = []

		# read in from tabix, may be empty
		W_k = tg.read_tabix(start, end, target=target, raise_error=False, **weights_info[k])

		# count number of non-zero snps for output to info file
		wk_n_snps.update({W(k)+'_N_SNP': W_k.index.size})

		while not W_k.empty:
			## CURRENTLY IGNORES ENSEMBLE VERSION; targets NOT guaranteed to be right target.version
			# ## CONSIDER CHECKING IF ('.' in target) and requiring ensemble version if specified in the gene annotation file; otherwise use whatever ensemble version in the weight file
			# if np.any(W_k.TargetID.str.contains('.')):
			# 	W_k['TargetID'] = W_k.TargetID.str.split('.', expand=True)[0]

			if not k == tabix_target_ks[0]:

				if not Weights.empty:
					W_k['snpIDflip'] = tg.get_snpIDs(W_k, flip=True)

					# snps that are neither in Weights nor are flipped versions of snps in Weights
					new_snpID = W_k['snpID'][np.invert(np.any(W_k[['snpID','snpIDflip']].isin(Weights.snpID.values), axis=1))]

					current_snpID = np.union1d(Weights.snpID, new_snpID)

					W_k = W_k[np.any(W_k[['snpID','snpIDflip']].isin(current_snpID), axis=1)].reset_index(drop=True)

					# if not in Geno.snpIDs, assumed flipped; if flipped, 1 - MAF and -ES
					flip = np.where(W_k.snpID.isin(current_snpID), True, False)

					if not np.all(flip):
						# set correct snpID, MAF, ES
						W_k['snpID'] = np.where(flip, W_k.snpID, W_k.snpIDflip)
						# if args.maf_diff:
						# 	W_k[MAF(k)] = np.where(flip, W_k[MAF(k)], 1 - W_k[MAF(k)])
						W_k[ES(k)] = np.where(flip, W_k[ES(k)], -W_k[ES(k)])

					W_k = W_k.drop(columns=['TargetID','snpIDflip'])

			target_ks.append(k)

			if Weights.empty:
				Weights = W_k.drop(columns=['TargetID'])
			else:
				Weights = Weights.merge(W_k, how='outer', on=['CHROM','POS','REF','ALT','snpID'])
			break

		else:
			empty_target_ks.append(k)

	if Weights.empty:
		print('No cis-eQTL weights with snp overlap in genotype data for target.\n')
		return None

	# ks may be added in read in step, need to sort for correct output later
	empty_target_ks.sort()

	# add 0 columns for weight files without data
	empty_wk_cols = [ES(k) for k in empty_target_ks]
	Weights[empty_wk_cols] = 0
	# Weights[empty_wk_cols] = np.nan

	# may include maf diff stuff later with W_0 as the reference
	# filter by MAF diff - if an MAF# differs too much, set ESk to nan
	# if args.maf_diff:
	# 	for k in range(Kweights):
	# 		maf_diff = np.abs(Train[MAF(k)].values - Train['MAF'].values)
	# 		Train.loc[maf_diff > args.maf_diff, ES(k)] = np.nan

	# filter out snps where all weights are nan; should not be possible to get all missing weights anywhere; since replaced with 0
	# all_missing_weights = Weights[ES_cols].count(axis=1) == 0
	# Weights = Weights[~all_missing_weights]
	n_snps = Weights.index.size

	if not n_snps:
		print('No valid SNPs.\n')
		return None

	# do stacked regression
	print('Averaging models.')
	Weights['ES'] = np.mean(np.nan_to_num(Weights[ES_cols].values), axis=1)
	
	# output Trained weights
	Weight_Out = Weights[['CHROM','POS','REF','ALT','snpID', 'ES']].copy()
	Weight_Out['ID'] = Weight_Out.snpID
	Weight_Out['TargetID'] = target
	# filter for non-zero effect size
	Weight_Out = Weight_Out[Weight_Out['ES'] != 0]
	# reorder columns
	Weight_Out = Weight_Out[out_weight_cols]
	Weight_Out = Weight_Out.astype(out_weight_dtypes)

	Weight_Out.to_csv(
		out_weight_path,
		sep='\t',
		header=None,
		index=None,
		mode='a')

	# output training info
	Info = Gene_Info[['CHROM','GeneStart','GeneEnd','TargetID','GeneName']].copy().reset_index()
	Info['N_SNP'] = n_snps
	Info['N_EFFECT_SNP'] = Weight_Out.ES.size
	# Info[info_wk_cols] = wk_n_snps

	# Info = Info.astype(info_wk_dtypes)
	Info = Info.join(pd.DataFrame.from_records(wk_n_snps, index=[0]).astype(info_wk_dtypes))[info_cols]

	Info = Info.astype(info_wk_dtypes)

	Info.to_csv(
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
	print('Done training for '+ str(n_targets) + ' target genes.\n')


###############################################################
# time calculation
elapsed_sec = time()-start_time
elapsed_time = tg.format_elapsed_time(elapsed_sec)
print('Computation time (DD:HH:MM:SS): ' + elapsed_time)

