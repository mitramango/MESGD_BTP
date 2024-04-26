#!/bin/bash


pip install gdown

fileid1="1oRNIIqTLdwBZh5N1tdZ_nZNODpSIDAsS"

fileid2="19YF_-YpyByP-Uj0efEfUvCmtSf7LHNVQ"

mkdir -p datasets

gdown --id ${fileid1} --output datasets/nq.gz
gdown --id ${fileid2} --output datasets/zsre.gz


zipfile1="datasets/nq" 
zipfile2="datasets/zsre"

gunzip ${zipfile1}
gunzip ${zipfile2}

tar -xvf datasets/nq -C datasets/
tar -xvf datasets/zsre -C datasets/