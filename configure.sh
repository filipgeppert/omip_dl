#! /bin/bash
# Script for instance configuration
# Assumption: we ran on ec2 with gpu pre configured
sudo apt-get update
source activate pytorch_p36
conda install pandas
source deactivate
