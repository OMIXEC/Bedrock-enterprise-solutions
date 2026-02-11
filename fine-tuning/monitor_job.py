#!/usr/bin/env python3
"""Monitor a Bedrock fine-tuning job.

Usage:
    python monitor_job.py --job-name my-finetune-job
    python monitor_job.py --job-name my-finetune-job --watch  # poll every 60s
    python monitor_job.py --job-name my-finetune-job --watch --interval 30
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError


# Status indicators for terminal output
STATUS_ICONS = {
    "InProgress": "[RUNNING]",
    "Completed": "[  DONE ]",
    "Failed": "[FAILED ]",
    "Stopping": "[STOPPING]",
    "Stopped": "[STOPPED]",
}

STATUS_COLORS = {
    "InProgress": "\033[33m",  # Yellow
    "Completed": "\033[32m",   # Green
    "Failed": "\033[31m",      # Red
    "Stopping": "\033[33m",    # Yellow
    "Stopped": "\033[90m",     # Gray
}
RESET_COLOR = "\033[0m"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor a Bedrock model fine-tuning job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status once
  python monitor_job.py --job-name my-finetune-job

  # Watch until completion (poll every 60s)
  python monitor_job.py --job-name my-finetune-job --watch

  # Custom poll interval
  python monitor_job.py --job-name my-finetune-job --watch --interval 30
        """,
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="Name or ARN of the fine-tuning job to monitor.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll until the job completes or fails.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds when --watch is enabled (default: 60).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region. Default: from AWS CLI configuration.",
    )
    return parser.parse_args()


def format_duration(seconds):
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_job_status(bedrock_client, job_name):
    """Fetch the current status of a fine-tuning job."""
    try:
        response = bedrock_client.get_model_customization_job(
            jobIdentifier=job_name
        )
        return response
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ResourceNotFoundException":
            print(f"ERROR: Job '{job_name}' not found.")
            print("  Check the job name or use the full job ARN.")
            print("  List all jobs: aws bedrock list-model-customization-jobs")
        elif error_code == "AccessDeniedException":
            print(f"ERROR: Access denied. Check your IAM permissions for bedrock:GetModelCustomizationJob.")
        else:
            print(f"ERROR: {error_code} - {e.response['Error']['Message']}")
        sys.exit(1)


def display_job_status(job_info, is_watch=False):
    """Pretty-print the job status information."""
    status = job_info.get("status", "Unknown")
    color = STATUS_COLORS.get(status, "")
    icon = STATUS_ICONS.get(status, f"[{status}]")

    now = datetime.now(timezone.utc)

    print()
    print("=" * 65)
    print(f"  Fine-Tuning Job: {job_info.get('jobName', 'N/A')}")
    print(f"  {color}{icon} Status: {status}{RESET_COLOR}")
    print("=" * 65)

    # Basic info
    print(f"  Job ARN:        {job_info.get('jobArn', 'N/A')}")
    print(f"  Base Model:     {job_info.get('baseModelIdentifier', 'N/A')}")
    print(f"  Custom Model:   {job_info.get('outputModelName', 'N/A')}")

    # Timing
    creation_time = job_info.get("creationTime")
    end_time = job_info.get("endTime")
    last_modified = job_info.get("lastModifiedTime")

    if creation_time:
        print(f"  Started:        {creation_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        elapsed = (end_time or now) - creation_time
        elapsed_seconds = elapsed.total_seconds()
        print(f"  Elapsed:        {format_duration(elapsed_seconds)}")

    if end_time:
        print(f"  Completed:      {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    elif creation_time and status == "InProgress":
        # Estimate remaining time (rough: most jobs take 1-4 hours)
        elapsed_min = (now - creation_time).total_seconds() / 60
        if elapsed_min > 5:
            print(f"  Estimated:      Fine-tuning jobs typically take 1-4 hours")

    # Hyperparameters
    hyper_params = job_info.get("hyperParameters", {})
    if hyper_params:
        print()
        print("  Hyperparameters:")
        for key, value in sorted(hyper_params.items()):
            print(f"    {key}: {value}")

    # Training data
    training_config = job_info.get("trainingDataConfig", {})
    if training_config:
        print()
        print(f"  Training Data:  {training_config.get('s3Uri', 'N/A')}")

    validation_config = job_info.get("validationDataConfig", {})
    validators = validation_config.get("validators", [])
    if validators:
        for v in validators:
            print(f"  Validation:     {v.get('s3Uri', 'N/A')}")

    output_config = job_info.get("outputDataConfig", {})
    if output_config:
        print(f"  Output:         {output_config.get('s3Uri', 'N/A')}")

    # Training metrics (available after completion)
    training_metrics = job_info.get("trainingMetrics", {})
    if training_metrics:
        print()
        print("  Training Metrics:")
        training_loss = training_metrics.get("trainingLoss")
        if training_loss is not None:
            print(f"    Training Loss:   {training_loss:.6f}")

    validation_metrics = job_info.get("validationMetrics", [])
    if validation_metrics:
        print("  Validation Metrics:")
        for metric in validation_metrics:
            val_loss = metric.get("validationLoss")
            if val_loss is not None:
                print(f"    Validation Loss: {val_loss:.6f}")

    # Output model ARN (available after completion)
    output_model_arn = job_info.get("outputModelArn")
    if output_model_arn:
        print()
        print(f"  Output Model ARN: {output_model_arn}")
        print()
        print("  Next steps:")
        print(f"    1. Create provisioned throughput:")
        print(f"       aws bedrock create-provisioned-model-throughput \\")
        print(f"         --model-id {output_model_arn} \\")
        print(f"         --provisioned-model-name my-custom-model \\")
        print(f"         --model-units 1")
        print()
        print(f"    2. Or evaluate the model:")
        print(f"       python evaluate.py --model-id {output_model_arn} \\")
        print(f"         --test-data data/validation.jsonl")

    # Failure reason
    failure_reason = job_info.get("failureMessage")
    if failure_reason:
        print()
        print(f"  {STATUS_COLORS['Failed']}Failure Reason: {failure_reason}{RESET_COLOR}")

    print()


def watch_job(bedrock_client, job_name, interval):
    """Continuously poll the job status until completion."""
    terminal_statuses = {"Completed", "Failed", "Stopped"}
    poll_count = 0

    print(f"Watching job '{job_name}' (polling every {interval}s, Ctrl+C to stop)...")

    try:
        while True:
            poll_count += 1
            job_info = get_job_status(bedrock_client, job_name)
            status = job_info.get("status", "Unknown")

            # Clear screen for clean output on each poll
            if poll_count > 1:
                print(f"\n--- Poll #{poll_count} at {datetime.now().strftime('%H:%M:%S')} ---")

            display_job_status(job_info, is_watch=True)

            if status in terminal_statuses:
                if status == "Completed":
                    print("Job completed successfully!")
                elif status == "Failed":
                    print("Job failed. Check the failure reason above.")
                    sys.exit(1)
                else:
                    print(f"Job reached terminal status: {status}")
                return

            print(f"Next check in {interval}s... (Ctrl+C to stop watching)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching. Job continues running in AWS.")
        print(f"Resume monitoring: python monitor_job.py --job-name {job_name} --watch")


def main():
    """Main entry point."""
    args = parse_args()
    region = args.region or boto3.Session().region_name
    bedrock_client = boto3.client("bedrock", region_name=region)

    if args.watch:
        watch_job(bedrock_client, args.job_name, args.interval)
    else:
        job_info = get_job_status(bedrock_client, args.job_name)
        display_job_status(job_info)


if __name__ == "__main__":
    main()
