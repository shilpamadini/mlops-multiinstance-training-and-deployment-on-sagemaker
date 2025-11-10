# Production-Grade Image Classification on AWS
## Project Overview

This project extends a previously completed image classification pipeline by preparing it for production-grade deployment using multiple AWS services. The model classifies dog breeds using a pretrained ResNet-50 architecture trained and deployed via Amazon SageMaker.
In this phase, the focus was on optimizing for scalability, efficiency, and real-world reliability through:

. Multi-instance distributed training on SageMaker

. EC2-based model training outside SageMaker

. Lambda-based inference endpoint integration

. Security, concurrency, and auto-scaling configuration

These steps represent the transition from a research prototype to a production-ready ML system.

## AWS SageMaker Notebook Instance Setup

A SageMaker notebook instance was created to run the training and deployment workflow.
I selected the instance type ml.t3.medium for initial orchestration because it provides sufficient compute for setup and data preparation at low cost and quick launch time.

![sagemakernotebookinstance](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/sagemaker-notebook-instance.png)

### Justification:
The ml.t3.medium instance balances cost-effectiveness with performance for non-intensive operations such as dataset upload, HPO orchestration, and deployment configuration.

## Data Preparation and Upload to S3

The dog breed classification dataset (dogImages.zip) was downloaded and extracted locally, then uploaded to a custom S3 bucket:
The dataset was organized into training, validation, and testing directories and used as the input channel for SageMaker training jobs.
```
wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
unzip dogImages.zip
aws s3 cp dogImages s3://mlops-sagemaker-imageclassification/ --recursive
```
![s3 bucket](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/s3-bucket.png)

![s3 bucket](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/data-added-s3.png)

## Single-Instance Training and Deployment

Initially, the train_and_deploy-solution.ipynb notebook trained and deployed the model using a single instance.
The configuration used:

Instance type: ml.m5.xlarge

Instance count: 1

Framework: PyTorch 1.4.0

Base job name: dog-pytorch

This served as a baseline to evaluate performance, cost, and training time before scaling to multi-instance distributed training.
![single-instance-training](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/estimator-single-instance.png)


## Transition to Multi-Instance Training

The same notebook was updated to enable multi-instance training by setting:

instance_count = 3
instance_type = 'ml.m5.xlarge'
This configuration allows each instance to have a full copy of the dataset for synchronous distributed learning, reducing communication overhead between nodes.
![multi-instance-training](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/estimator-multi-instnace.png)

Training completed in approximately the same time as the single-instance setup (~19 minutes), indicating that the dataset size and network bandwidth were not large enough for a significant speedup. However, the setup validated multi-node readiness and fault-tolerant distributed configuration.

## EC2-Based Model Training

To demonstrate flexibility and portability, the model training code was adapted to run directly on an EC2 instance using the ec2train1.py script.

EC2 Instance Selected:
g4dn.xlarge running Deep Learning OSS Nvidia Driver AMI (Amazon Linux 2023) with PyTorch 2.8 pre-installed.

### Justification:
This instance provides a cost-effective GPU environment suitable for fine-tuning CNN-based models.
Instead of manually installing dependencies, I activated the pre-built PyTorch environment:

```
source /opt/pytorch/bin/activate
```
This approach minimized setup time and leveraged CUDA acceleration without additional configuration.

![ec21](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/Ec2-1.png)

![ec22](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/ec2-2.png)

![ec23](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/ec2-3.png)

![ec24](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/ec2-4.png)

![ec25](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/ec2-saved-model.png)

### Differences from SageMaker training:

Manual control of data management and checkpointing

No built-in monitoring or managed storage integration

SageMaker modules like sagemaker.debugger and sagemaker.tuner were omitted

The EC2 script (ec2train1.py) reused modular functions (train/test/net) from hpo.py for consistency

This highlights how code can be adapted for local or external compute while maintaining portability and reproducibility.

## Lambda Function Setup

A Lambda function (lambdafunction.py) was configured to invoke the SageMaker endpoint for inference.
The function:

Loads an input image URL (passed in event JSON)

Calls the deployed endpoint using the invoke_endpoint() method

Returns predicted class probabilities

Key modification:

endpoint_name = "dog-pytorch-2025-11-09-23-47-58-682"  # actual deployed endpoint name


How it works:
The Lambda function uses the boto3 client for SageMaker runtime, sends a serialized image payload to the endpoint, and receives model predictions as a JSON response.

Example test event:

{
  "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg"
}


The output was a JSON array of 133 probabilities corresponding to dog breed classes.

![ec25](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/e88b484c0f5435c69f91d7df8cb8aea5e76d3e4a/images/ec2-saved-model.png)

