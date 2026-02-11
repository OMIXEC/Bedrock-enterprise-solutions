#!/usr/bin/env python3
"""Start a Bedrock model fine-tuning job.

Usage:
    python run_job.py --model amazon.nova-micro-v1:0 --training-data s3://bucket/training.jsonl
    python run_job.py --model amazon.nova-micro-v1:0 --bucket my-bucket  # uploads local data first
    python run_job.py --model amazon.nova-micro-v1:0 --training-data data/training.jsonl --bucket my-bucket
"""

import argparse
import json
import os
import sys
import time
import uuid

import boto3
from botocore.exceptions import ClientError


SUPPORTED_MODELS = [
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.titan-text-express-v1",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "cohere.command-r-v1:0",
]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Start a Bedrock model fine-tuning (customization) job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing S3 data
  python run_job.py --model amazon.nova-micro-v1:0 \\
      --training-data s3://my-bucket/training.jsonl

  # Upload local data first
  python run_job.py --model amazon.nova-micro-v1:0 \\
      --training-data data/training.jsonl \\
      --bucket my-finetuning-bucket

  # Full customization
  python run_job.py --model amazon.nova-micro-v1:0 \\
      --training-data s3://bucket/train.jsonl \\
      --validation-data s3://bucket/val.jsonl \\
      --job-name mfg-qc-v2 \\
      --epochs 5 --batch-size 4 --learning-rate 0.00005
        """,
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Base model ID. Supported: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--training-data",
        required=True,
        help="S3 URI (s3://...) or local path to training JSONL file.",
    )
    parser.add_argument(
        "--validation-data",
        default=None,
        help="S3 URI or local path to validation JSONL file (optional).",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket for uploading local data and storing output. Required if training-data is a local path.",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Name for the fine-tuning job. Default: auto-generated.",
    )
    parser.add_argument(
        "--custom-model-name",
        default=None,
        help="Name for the resulting custom model. Default: derived from job name.",
    )
    parser.add_argument(
        "--role-arn",
        default=None,
        help="IAM role ARN for Bedrock fine-tuning. Auto-detected if not provided.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region. Default: from AWS CLI configuration.",
    )
    return parser.parse_args()


def validate_jsonl_file(filepath):
    """Validate that a local file is valid JSONL format for Bedrock fine-tuning."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    line_count = 0
    errors = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                line_count += 1
                # Check for expected fields (Bedrock format)
                if "prompt" not in record and "messages" not in record and "system" not in record:
                    errors.append(
                        f"  Line {i}: Missing 'prompt' or 'messages' field."
                    )
            except json.JSONDecodeError as e:
                errors.append(f"  Line {i}: Invalid JSON - {e}")

    if errors:
        print(f"WARNING: Found {len(errors)} issue(s) in {filepath}:")
        for err in errors[:5]:
            print(err)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more.")
    else:
        print(f"Validated {filepath}: {line_count} records, format OK.")

    if line_count == 0:
        print(f"ERROR: No valid JSONL records found in {filepath}")
        sys.exit(1)

    return line_count


def upload_to_s3(local_path, bucket, s3_key, s3_client):
    """Upload a local file to S3 and return the S3 URI."""
    try:
        print(f"Uploading {local_path} -> s3://{bucket}/{s3_key} ...")
        s3_client.upload_file(local_path, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"Upload complete: {s3_uri}")
        return s3_uri
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchBucket":
            print(f"ERROR: Bucket '{bucket}' does not exist.")
            print(f"  Create it first: aws s3 mb s3://{bucket}")
        elif error_code == "AccessDenied":
            print(f"ERROR: Access denied to bucket '{bucket}'.")
            print("  Check your IAM permissions for s3:PutObject.")
        else:
            print(f"ERROR: S3 upload failed: {e}")
        sys.exit(1)


def resolve_data_uri(data_path, bucket, prefix, s3_client):
    """Resolve a data path to an S3 URI, uploading if necessary."""
    if data_path is None:
        return None

    # Already an S3 URI
    if data_path.startswith("s3://"):
        return data_path

    # Local file -- needs upload
    if bucket is None:
        print(f"ERROR: --bucket is required when using a local file path ({data_path}).")
        print("  Provide --bucket <bucket-name> to upload local data to S3.")
        sys.exit(1)

    validate_jsonl_file(data_path)
    filename = os.path.basename(data_path)
    s3_key = f"{prefix}/{filename}"
    return upload_to_s3(data_path, bucket, s3_key, s3_client)


def find_finetuning_role(iam_client):
    """Try to find an existing Bedrock fine-tuning role."""
    role_names_to_try = [
        "BedrockFineTuningRole",
        "AmazonBedrockFineTuningRole",
        "bedrock-finetuning-role",
    ]
    for role_name in role_names_to_try:
        try:
            response = iam_client.get_role(RoleName=role_name)
            return response["Role"]["Arn"]
        except ClientError:
            continue
    return None


def create_finetuning_job(args):
    """Create and start a Bedrock model customization job."""
    region = args.region or boto3.Session().region_name
    bedrock_client = boto3.client("bedrock", region_name=region)
    s3_client = boto3.client("s3", region_name=region)
    iam_client = boto3.client("iam", region_name=region)

    # Validate model
    if args.model not in SUPPORTED_MODELS:
        print(f"WARNING: Model '{args.model}' is not in the known supported list.")
        print(f"  Supported models: {', '.join(SUPPORTED_MODELS)}")
        print("  Proceeding anyway -- the API will reject if truly unsupported.\n")

    # Generate job name if not provided
    short_id = uuid.uuid4().hex[:6]
    job_name = args.job_name or f"finetune-{short_id}"
    custom_model_name = args.custom_model_name or f"custom-{job_name}"

    print("=" * 60)
    print("Bedrock Fine-Tuning Job Configuration")
    print("=" * 60)
    print(f"  Base model:    {args.model}")
    print(f"  Job name:      {job_name}")
    print(f"  Custom model:  {custom_model_name}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Region:        {region}")
    print()

    # Resolve training data URI
    prefix = f"fine-tuning/{job_name}"
    training_uri = resolve_data_uri(
        args.training_data, args.bucket, prefix, s3_client
    )
    print(f"  Training data: {training_uri}")

    # Resolve validation data URI
    validation_config = None
    if args.validation_data:
        validation_uri = resolve_data_uri(
            args.validation_data, args.bucket, prefix, s3_client
        )
        print(f"  Validation:    {validation_uri}")
        validation_config = {
            "validators": [{"s3Uri": validation_uri}]
        }

    # Output location
    if args.bucket:
        output_bucket = args.bucket
    else:
        # Extract bucket from training URI
        output_bucket = training_uri.replace("s3://", "").split("/")[0]
    output_uri = f"s3://{output_bucket}/{prefix}/output/"
    print(f"  Output:        {output_uri}")

    # Resolve IAM role
    role_arn = args.role_arn
    if not role_arn:
        role_arn = find_finetuning_role(iam_client)
        if not role_arn:
            print("\nERROR: No --role-arn provided and could not auto-detect a Bedrock fine-tuning role.")
            print("  Create a role with Bedrock and S3 permissions, then pass --role-arn <arn>.")
            print("  See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-iam-role.html")
            sys.exit(1)
    print(f"  Role ARN:      {role_arn}")
    print()

    # Build job request
    job_params = {
        "jobName": job_name,
        "customModelName": custom_model_name,
        "roleArn": role_arn,
        "baseModelIdentifier": args.model,
        "customizationType": "FINE_TUNING",
        "trainingDataConfig": {"s3Uri": training_uri},
        "outputDataConfig": {"s3Uri": output_uri},
        "hyperParameters": {
            "epochCount": str(args.epochs),
            "batchSize": str(args.batch_size),
            "learningRate": str(args.learning_rate),
        },
    }

    if validation_config:
        job_params["validationDataConfig"] = validation_config

    # Submit the job
    print("Submitting fine-tuning job...")
    try:
        response = bedrock_client.create_model_customization_job(**job_params)
        job_arn = response["jobArn"]
        print()
        print("=" * 60)
        print("Job submitted successfully!")
        print("=" * 60)
        print(f"  Job ARN: {job_arn}")
        print(f"  Job Name: {job_name}")
        print()
        print("Monitor progress:")
        print(f"  python monitor_job.py --job-name {job_name}")
        print(f"  python monitor_job.py --job-name {job_name} --watch")
        print()
        print("Or via AWS CLI:")
        print(f"  aws bedrock get-model-customization-job --job-identifier {job_name}")
        print()
        return job_arn

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]

        if error_code == "ValidationException":
            print(f"\nERROR: Validation failed: {error_msg}")
            if "model" in error_msg.lower():
                print("  The base model may not support fine-tuning in this region.")
                print(f"  Try a different model. Supported: {', '.join(SUPPORTED_MODELS)}")
            elif "data" in error_msg.lower() or "format" in error_msg.lower():
                print("  Check your training data format.")
                print("  Expected JSONL with 'prompt'/'completion' or 'messages' fields.")
        elif error_code == "ResourceNotFoundException":
            print(f"\nERROR: Resource not found: {error_msg}")
            print("  Verify the S3 bucket and data paths exist.")
        elif error_code == "AccessDeniedException":
            print(f"\nERROR: Access denied: {error_msg}")
            print("  Ensure the IAM role has bedrock:CreateModelCustomizationJob permission.")
        elif error_code == "ServiceQuotaExceededException":
            print(f"\nERROR: Quota exceeded: {error_msg}")
            print("  You may have reached the concurrent fine-tuning job limit.")
            print("  Wait for existing jobs to complete or request a quota increase.")
        elif error_code == "TooManyRequestsException":
            print(f"\nERROR: Too many requests: {error_msg}")
            print("  Wait a moment and try again.")
        else:
            print(f"\nERROR: {error_code} - {error_msg}")

        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    create_finetuning_job(args)
