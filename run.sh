#! /bin/bash
# Script for model training and result transfer to s3
# Activate conda env
source activate pytorch_p36
python manage.py
# copy results to s3
