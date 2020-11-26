#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output
mkdir -p test_dir/mlruns

rm test_dir/model/*
rm test_dir/output/*
rm test_dir/mlruns/*

docker run -v $(pwd)/test_dir:/opt/ml  -v /home/ec2-user/SageMaker/Accessbank_CTR/Catboost-sagemaker/container/local_test/test_dir/mlruns:/opt/program/mlruns --rm ${image} train


