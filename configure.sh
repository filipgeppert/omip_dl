#! /bin/bash
# Script for instance configuration
# Assumption: we ran on ec2 with gpu pre configured
TOP_DIRECTORY="omip_dl"
BUCKET_NAME="omip-images"

sudo apt-get update
git clone https://github.com/filipgeppert/omip_dl.git
aws s3 cp s3://$BUCKET_NAME/images/ ~/$TOP_DIRECTORY/images/ --recursively
source activate pytorch_p36
conda install pandas
source deactivate
