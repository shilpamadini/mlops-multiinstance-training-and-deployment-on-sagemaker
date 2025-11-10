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

After completing single-instance training, I configured multi-instance training using three ml.m5.xlarge instances.
The data distribution strategy was kept at the default setting (SageMaker’s default behavior for non-specified channels, typically FullyReplicated for File mode). 

The intent was to observe scaling effects when moving from a single GPU/CPU node to a small cluster configuration.
The multi-instance job ran successfully in about 21 minutes, compared to 19 minutes for single-instance training, showing minimal performance improvement.Since the dataset fits easily in memory, reducing the benefit of data parallelism.

## EC2-Based Model Training

To demonstrate flexibility and portability, the model training code was adapted to run directly on an EC2 instance using the ec2train1.py script.

I chose the Deep Learning OSS Nvidia Driver AMI – GPU PyTorch 2.8 (Amazon Linux 2023) on a g4dn.xlarge instance because it provides a pre-configured environment optimized for GPU-based deep learning workloads. The instance includes an NVIDIA T4 GPU, 4 vCPUs, and 16 GB of memory, which is a good balance between cost and performance for computer vision tasks such as ResNet-50 image classification. The GPU enables accelerated tensor operations and significantly faster training compared to CPU-only instances. The Amazon Linux 2023 base image is secure and lightweight, and the AMI is officially maintained by AWS, ensuring stable CUDA, cuDNN, and PyTorch integrations.
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

## Model Endpoint Deployment

After successful training, the model artifact was deployed as a real-time inference endpoint directly from SageMaker using:

```
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
```

This automatically created a managed HTTPS endpoint visible under Inference > Endpoints in the AWS Console.

The endpoint could handle inference requests through the SageMaker SDK or external services via the AWS API.

### Why deploy through SageMaker:

Provides fully managed infrastructure for hosting ML models.

Handles load balancing, scaling, and security automatically.

Offers seamless integration with Lambda and API Gateway for downstream applications.

Enables monitoring through CloudWatch metrics.

The deployed endpoint name was recorded for use by the Lambda function:

```
dog-pytorch-2025-11-09-23-47-58-682
```
## Lambda Function Setup

The lambdafunction.py script serves as an integration bridge between the deployed SageMaker endpoint and external applications via AWS Lambda.  The function uses Python 3.x runtime and leverages the boto3 SDK to create a SageMaker runtime client that invokes the endpoint using the invoke_endpoint() method. When an inference request (such as an image in Base64-encoded format) is sent to the Lambda function, it forwards the request payload to the SageMaker endpoint, waits for the model’s prediction, and then returns the inference result as a JSON response. The invoke_endpoint() method is critical here as it sends the request to SageMaker’s REST API and retrieves the model output in real-time. This setup makes the model accessible via an API-style call, enabling integration with web apps, data pipelines, or other AWS services like API Gateway. The return statement is structured to handle status codes and formatted responses, ensuring smooth interaction between clients and the model service.

Example test event:

```
{
  "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg"
}
```

The output was a JSON array of 133 probabilities corresponding to dog breed classes.

![lamda1](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/lamda-function.png)


## Security

In reviewing security across my AWS workspace, I identified that several roles and policies granted FullAccess to SageMaker, Lambda, and S3. While this was acceptable for development and debugging, it would not be ideal in production. To improve security, I would replace broad policies with least-privilege policies that only allow access to the specific endpoint, S3 paths, and logs required for inference. Additionally, I would routinely audit IAM roles to remove old, unused, or inactive ones, as these could pose unnecessary exposure risks.

To allow my Lambda function to successfully invoke the deployed SageMaker endpoint, I first located the Lambda execution role in the IAM console. By default, each Lambda function is assigned an IAM role that defines what AWS services it can access. I attached the AmazonSageMakerFullAccess policy to this role so the Lambda function could invoke any SageMaker endpoint. This permission is necessary for the boto3 client within the function to execute the invoke_endpoint() API call. However, in a production environment, a more secure and recommended approach would be to attach a custom policy that only grants permission to invoke a specific endpoint, using the ARN of that endpoint instead of a full-access policy.

![iamroles](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/IAM-roles-dashboard.png)

![iamroles](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/iam-lamda-sagemaker-full-access.png)


Security Considerations:

Applied least privilege principle (only SageMaker access granted)



Example lamda test event:
```
{
  "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/20113314/Carolina-Dog-standing-outdoors.jpg"
}
```

Test Output:
The Lambda function returned a list of probabilities (133 values) representing breed predictions for the input image.

![lamda2](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/lamda-test-successful.png)

## Concurrency and Auto-Scaling Configuration

After deploying the Lambda function, I configured Provisioned Concurrency = 4 to maintain four pre-initialized instances of the function. This setup ensures low-latency responses and eliminates cold starts during inference. The choice of four was made to provide adequate throughput for parallel inference requests while keeping costs manageable for a production-like workload. This configuration allows the Lambda function to handle multiple concurrent invocations efficiently without scaling delays.

For the SageMaker endpoint, I enabled Auto Scaling using the Inference > Endpoints section of the AWS Console. The configuration was as follows:

Minimum instance count: 2

Maximum instance count: 4

Target value: 10 invocations per instance per minute

Scale-in cooldown: 30 seconds

Scale-out cooldown: 30 seconds

This ensures that SageMaker dynamically adjusts compute capacity based on actual inference traffic. The cooldown settings prevent rapid oscillations in instance counts, allowing the system to stabilize between scale events

![concurrency](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/lamda-version.png)

![concurrency](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/provisioned-concurrency.png)

![autoscaling](https://github.com/shilpamadini/mlops-multiinstance-training-and-deployment-on-sagemaker/blob/469911c1194e13ec18a693bd245e9899b63acc12/images/autoscaling-endpoint.png)


S3 bucket retained only for dataset storage



