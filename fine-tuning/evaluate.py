#!/usr/bin/env python3
"""Evaluate a fine-tuned model against test data.

Usage:
    python evaluate.py --model-id arn:aws:bedrock:us-east-1:123456789:custom-model/my-model \\
        --test-data data/validation.jsonl
    python evaluate.py --model-id arn:aws:bedrock:... --test-data validation.jsonl --output results.json
    python evaluate.py --model-id arn:aws:bedrock:... --test-data validation.jsonl --max-samples 5
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Bedrock model against test data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all validation examples
  python evaluate.py --model-id arn:aws:bedrock:us-east-1:123:custom-model/my-model \\
      --test-data data/validation.jsonl

  # Evaluate first 5 examples only
  python evaluate.py --model-id arn:aws:bedrock:... --test-data data/validation.jsonl --max-samples 5

  # Save results to file
  python evaluate.py --model-id arn:aws:bedrock:... --test-data data/validation.jsonl \\
      --output eval_results.json
        """,
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ARN or provisioned model ARN for the fine-tuned model.",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to JSONL test data file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens in model response (default: 1024).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region. Default: from AWS CLI configuration.",
    )
    return parser.parse_args()


def load_test_data(filepath, max_samples=None):
    """Load test data from a JSONL file."""
    if not os.path.exists(filepath):
        print(f"ERROR: Test data file not found: {filepath}")
        sys.exit(1)

    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                samples.append(record)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping invalid JSON at line {i + 1}: {e}")

    if max_samples:
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} test samples from {filepath}")
    return samples


def extract_prompt_and_expected(record):
    """Extract prompt and expected response from a test record.

    Supports two formats:
    1. {"prompt": "...", "completion": "..."} -- classic format
    2. {"system": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    if "prompt" in record:
        return record["prompt"], record.get("completion", "")

    if "messages" in record:
        messages = record["messages"]
        user_msg = None
        assistant_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]
        return user_msg, assistant_msg or ""

    return None, None


def invoke_model(bedrock_runtime, model_id, prompt, system_prompt=None, max_tokens=1024):
    """Invoke the fine-tuned model using the Converse API."""
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    kwargs = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.1,
        },
    }

    if system_prompt:
        kwargs["system"] = [{"text": system_prompt}]

    start_time = time.time()
    try:
        response = bedrock_runtime.converse(**kwargs)
        latency = time.time() - start_time

        output_text = ""
        output_message = response.get("output", {}).get("message", {})
        for block in output_message.get("content", []):
            if "text" in block:
                output_text += block["text"]

        usage = response.get("usage", {})
        return {
            "response": output_text,
            "latency_seconds": round(latency, 3),
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "stop_reason": response.get("stopReason", "unknown"),
        }
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg = e.response["Error"]["Message"]
        return {
            "response": None,
            "latency_seconds": time.time() - start_time,
            "error": f"{error_code}: {error_msg}",
            "input_tokens": 0,
            "output_tokens": 0,
        }


def compute_keyword_accuracy(expected, actual):
    """Compute keyword overlap accuracy between expected and actual responses.

    Extracts key terms from expected response and checks how many appear in actual.
    This is a simple heuristic; production systems should use domain-specific metrics.
    """
    if not expected or not actual:
        return 0.0

    # Extract significant words (longer than 3 chars, not common stopwords)
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "with",
        "this", "that", "from", "they", "been", "said", "each", "which",
        "their", "will", "other", "about", "many", "then", "them", "these",
        "some", "would", "make", "like", "into", "time", "very", "when",
        "come", "could", "more", "also",
    }

    def extract_keywords(text):
        words = set()
        for word in text.lower().split():
            # Clean punctuation
            cleaned = "".join(c for c in word if c.isalnum() or c == "-")
            if len(cleaned) > 3 and cleaned not in stopwords:
                words.add(cleaned)
        return words

    expected_keywords = extract_keywords(expected)
    actual_keywords = extract_keywords(actual)

    if not expected_keywords:
        return 1.0

    matched = expected_keywords & actual_keywords
    return len(matched) / len(expected_keywords)


def evaluate_model(args):
    """Run evaluation on all test samples."""
    region = args.region or boto3.Session().region_name
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

    # Load test data
    samples = load_test_data(args.test_data, args.max_samples)

    if not samples:
        print("ERROR: No test samples loaded.")
        sys.exit(1)

    print()
    print("=" * 65)
    print(f"  Evaluating: {args.model_id}")
    print(f"  Test data:  {args.test_data}")
    print(f"  Samples:    {len(samples)}")
    print("=" * 65)
    print()

    results = []
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    errors = 0
    accuracies = []

    for i, sample in enumerate(samples, 1):
        prompt, expected = extract_prompt_and_expected(sample)
        system_prompt = sample.get("system", None)

        if prompt is None:
            print(f"  [{i}/{len(samples)}] SKIP -- no prompt found in record")
            continue

        print(f"  [{i}/{len(samples)}] Evaluating...", end=" ", flush=True)

        result = invoke_model(
            bedrock_runtime, args.model_id, prompt,
            system_prompt=system_prompt,
            max_tokens=args.max_tokens,
        )

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            errors += 1
            result["sample_index"] = i
            result["prompt"] = prompt[:100] + "..." if len(prompt) > 100 else prompt
            results.append(result)
            continue

        # Compute accuracy
        accuracy = compute_keyword_accuracy(expected, result["response"])
        accuracies.append(accuracy)

        total_latency += result["latency_seconds"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        response_len = len(result["response"])
        print(
            f"OK  latency={result['latency_seconds']:.1f}s  "
            f"tokens={result['input_tokens']}+{result['output_tokens']}  "
            f"accuracy={accuracy:.0%}  "
            f"len={response_len}"
        )

        result["sample_index"] = i
        result["prompt"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
        result["expected"] = expected[:200] + "..." if expected and len(expected) > 200 else expected
        result["keyword_accuracy"] = round(accuracy, 4)
        results.append(result)

    # Compute aggregate metrics
    successful = len(results) - errors
    avg_latency = total_latency / successful if successful > 0 else 0
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    avg_response_length = (
        sum(len(r.get("response", "") or "") for r in results if not r.get("error"))
        / successful
        if successful > 0
        else 0
    )

    summary = {
        "model_id": args.model_id,
        "test_data": args.test_data,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_samples": len(samples),
        "successful": successful,
        "errors": errors,
        "metrics": {
            "avg_keyword_accuracy": round(avg_accuracy, 4),
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_response_length_chars": round(avg_response_length, 0),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        "results": results,
    }

    # Print summary table
    print()
    print("=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Model:              {args.model_id}")
    print(f"  Samples:            {len(samples)} total, {successful} successful, {errors} errors")
    print(f"  Keyword Accuracy:   {avg_accuracy:.1%}")
    print(f"  Avg Latency:        {avg_latency:.2f}s")
    print(f"  Avg Response Len:   {avg_response_length:.0f} chars")
    print(f"  Total Input Tokens: {total_input_tokens:,}")
    print(f"  Total Output Tokens:{total_output_tokens:,}")
    print(f"  Total Tokens:       {total_input_tokens + total_output_tokens:,}")
    print("=" * 65)

    # Accuracy distribution
    if accuracies:
        buckets = {"90-100%": 0, "70-89%": 0, "50-69%": 0, "<50%": 0}
        for acc in accuracies:
            if acc >= 0.9:
                buckets["90-100%"] += 1
            elif acc >= 0.7:
                buckets["70-89%"] += 1
            elif acc >= 0.5:
                buckets["50-69%"] += 1
            else:
                buckets["<50%"] += 1
        print()
        print("  Accuracy Distribution:")
        for bucket, count in buckets.items():
            bar = "#" * count
            print(f"    {bucket:>8}: {count:>3} {bar}")
    print()

    # Save results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Results saved to: {args.output}")
    else:
        print("Tip: Use --output results.json to save detailed results.")

    return summary


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
