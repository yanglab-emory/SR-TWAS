#!/usr/bin/bash

#######################################################################
### Input Arguments for Naive method
#######################################################################

###############################################################

# deprecated arguments
thread=0


# read arguments
weights=( )
weights_names=( )

while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        if [[ $v == "weights" ]]; then
            while [[ $2 != *"--"* ]]; do
                if [[ $2 == "" ]]; then 
                    break;
                fi
                weights+=("$2")
                shift
            done
        elif [[ $v == "weights_names" ]]; then
            while [[ $2 != *"--"* ]]; do
                if [[ $2 == "" ]]; then 
                    break;
                fi
                weights_names+=("$2")
                shift
            done    
        else
            declare $v="$2"
        fi
    fi
    shift
done

########## Set default value
# warn/set deprecated
if [[ "$thread"x != "0"x ]];then
    echo 'Warning: --thread is deprecated; Please use --parallel'
    parallel=${parallel:-${thread}}
else
    parallel=${parallel:-1}
fi

# set defaults
cvR2=${cvR2:-1}
cvR2_threshold=${cvR2_threshold:-0.005}
format=${format:-"GT"}
hwe=${hwe:-0.00001}
maf_diff=${maf_diff:-0}
missing_rate=${missing_rate:-0.2}
sub_dir=${sub_dir:-1}
weight_threshold=${weight_threshold:-0}
window=${window:-$((10**6))}

# output file names
out_prefix=${out_prefix:-CHR${chr}_naive_train}
out_weight_file=${out_weight_file:-${out_prefix}_eQTLweights.txt}
out_info_file=${out_info_file:-${out_prefix}_GeneInfo.txt}
log_file=${log_file:-${out_prefix}_log.txt}

# sub directory in out directory
if [[ "$sub_dir"x == "1"x ]];then
    out_sub_dir=${out_dir}/Naive_CHR${chr}
else
    out_sub_dir=${out_dir}
fi

############# TWAS 

## make output directory
mkdir -p ${out_dir}
mkdir -p ${out_dir}/logs
mkdir -p ${out_sub_dir}

# check tabix command
if [ ! -x "$(command -v tabix)" ]; then
    echo 'Error: required tool TABIX is not available.' >&2
    exit 1
fi

# Check gene expression file
if [ ! -f "${gene_exp}" ] ; then
    echo Error: Gene expression file ${gene_exp} does not exist or is empty. >&2
    exit 1
fi

# Check training sample ID file
if [ ! -f "${train_sampleID}" ]; then
    echo Error: Training sample ID file ${train_sampleID} does not exist or is empty. >&2
    exit 1
fi

# Check genotype file 
if [ ! -f "${genofile}" ]; then
    echo Error: Training genotype file ${genofile} does not exist or is empty. >&2
    exit 1
fi

## Naive
if [[ ! -x  ${SR_TWAS_dir}/Naive.py ]] ; then
    chmod 755 ${SR_TWAS_dir}/Naive.py
fi

python ${SR_TWAS_dir}/Naive.py \
--chr ${chr} \
--cvR2 ${cvR2} \
--cvR2_threshold ${cvR2_threshold} \
--format ${format} \
--gene_exp ${gene_exp} \
--genofile ${genofile} \
--genofile_type ${genofile_type} \
--hwe ${hwe} \
--maf_diff ${maf_diff} \
--missing_rate ${missing_rate} \
--out_dir ${out_sub_dir} \
--out_info_file ${out_info_file} \
--out_weight_file ${out_weight_file} \
--SR_TWAS_dir ${SR_TWAS_dir} \
--parallel ${parallel} \
--train_sampleID ${train_sampleID} \
--weight_threshold ${weight_threshold} \
--weights ${weights[@]} \
--weights_names ${weights_names[@]} \
--window ${window} \
> ${out_dir}/logs/${log_file}

echo "Completed SR-TWAS training."


# SORT, BGZIP, AND TABIX

# set temp, weight filepaths for sorting
temp=${out_sub_dir}/temp_${out_weight_file}
weight=${out_sub_dir}/${out_weight_file}

echo "Sort/bgzip/tabix-ing output weight file."
head -n1 ${temp} > ${weight}

tail -n+2 ${temp} | \
sort -nk1 -nk2 >> ${weight} && \
rm ${temp} || \
( tail -n+2 ${temp} | \
sort -T ${out_sub_dir} -nk1 -nk2 >> ${weight} && \
rm ${temp} )

if [ ! -f "${temp}" ] ; then
    echo "Sort successful. Bgzip/tabix-ing."
    bgzip -f ${weight} && \
    tabix -f -b2 -e2 -S1 ${weight}.gz

else
    echo "Sort failed; Unable to bgzip/tabix output weights file."
fi






