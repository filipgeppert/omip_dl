#! /bin/bash
# Script for instance configuration
# Assumption: we ran on ec2 with gpu pre configured
TOP_DIRECTORY="omip_dl"
BUCKET_NAME="omip-images"

sudo apt-get update
aws s3 cp --recursive s3://$BUCKET_NAME/images/ ~/$TOP_DIRECTORY/images/
source activate pytorch_p36
conda install pandas
source deactivate
