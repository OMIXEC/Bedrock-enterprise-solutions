# NY-Bank Amazon Bedrock Agent

Step-by-step implementation guide for dev teams (S3, DynamoDB, Lambda, Bedrock Agent, Action Groups/OpenAPI, Knowledge Base, Guardrails)

This document is a complete build guide to implement a retail-banking Bedrock Agent for NY-Bank that can answer “new account status” questions by calling a Lambda action (backed by DynamoDB) and optionally using a Knowledge Base (RAG). It also includes a guardrail template and production hardening steps.

---

## 0. Target outcome

The NY-Bank agent can:

* Accept a user request like: “What is the status of account 5555?”
* Use an Action Group (OpenAPI schema) to call Lambda
* Lambda calls DynamoDB `customerAccountStatus` by `AccountID`
* Return a structured response to the agent
* The agent responds politely, without hallucinating, and refuses out-of-scope requests
* Optionally uses a Knowledge Base for general policy/FAQ text

---

## 1. Prerequisites

### 1.1 AWS account and region

Pick one region for all resources (recommended: same region for Bedrock Agent, Lambda, DynamoDB, S3, Knowledge Base). Ensure Bedrock model access is enabled in that region.

### 1.2 Local tooling

* AWS CLI configured (`aws configure`)
* Permissions to create: IAM roles/policies, DynamoDB, Lambda, S3, Bedrock Agent, Knowledge Base resources

---

## 2. Create S3 buckets (OpenAPI schema and Knowledge Base)

You will use two buckets:

* `nybank-agent-openapi` (stores OpenAPI schema for action groups)
* `nybank-agent-knowledgebase` (stores documents for Knowledge Base)

### 2.1 Create buckets

Example (replace region):

```bash
aws s3api create-bucket \
  --bucket nybank-agent-openapi \
  --region us-east-1

aws s3api create-bucket \
  --bucket nybank-agent-knowledgebase \
  --region us-east-1
```

### 2.2 Security baseline for both buckets

Recommended:

* Block Public Access: ON
* Default encryption: SSE-S3 or SSE-KMS
* Versioning: ON

Example:

```bash
aws s3api put-public-access-block \
  --bucket nybank-agent-openapi \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-versioning \
  --bucket nybank-agent-openapi \
  --versioning-configuration Status=Enabled
```

Repeat for `nybank-agent-knowledgebase`.

---

## 3. Create DynamoDB table `customerAccountStatus`

### 3.1 Table definition

* Table name: `customerAccountStatus`
* Partition key: `AccountID` (Number)
* Billing mode: On-Demand
* Encryption: enabled
* PITR: enabled (recommended)

CLI example:

```bash
aws dynamodb create-table \
  --table-name customerAccountStatus \
  --attribute-definitions AttributeName=AccountID,AttributeType=N \
  --key-schema AttributeName=AccountID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

Enable PITR:

```bash
aws dynamodb update-continuous-backups \
  --table-name customerAccountStatus \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

### 3.2 Insert sample records

Example:

```bash
aws dynamodb put-item \
  --table-name customerAccountStatus \
  --item '{
    "AccountID": {"N": "5555"},
    "CustomerName": {"S": "John Doe"},
    "AccountStatus": {"S": "Active"},
    "AccountType": {"S": "Checking"},
    "CreatedDate": {"S": "2025-12-01"}
  }'
```

---

## 4. Create IAM roles and policies

You need:

1. Lambda execution role (read DynamoDB + CloudWatch logs)
2. Bedrock Agent role (invoke Lambda action group + (optional) access KB)
3. Knowledge Base execution role (if KB requires it, depending on setup)

### 4.1 Lambda execution role (least privilege)

Attach:

* CloudWatch Logs basic execution
* DynamoDB GetItem on `customerAccountStatus`

Policy example (DynamoDB read):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["dynamodb:GetItem"],
      "Resource": "arn:aws:dynamodb:*:*:table/customerAccountStatus"
    }
  ]
}
```

Also attach the managed policy:

* `AWSLambdaBasicExecutionRole`

### 4.2 Bedrock Agent role (minimum needed)

For action groups:

* permission to invoke the Lambda function

Option A (recommended): Lambda resource-based policy granting `bedrock.amazonaws.com` permission (done later in Step 8).
Option B: IAM permission on the agent role (may still require Lambda permission depending on configuration).

For Knowledge Base:

* configure per your KB backend (OpenSearch Serverless / Aurora / etc.). The KB wizard typically creates or suggests required roles and policies.

---

## 5. Create Lambda function `newBankAccountStatus`

### 5.1 Create function

* Name: `newBankAccountStatus`
* Runtime: Python 3.12
* Timeout: 10 seconds
* Memory: 256 MB
* Role: Lambda execution role from Step 4.1

### 5.2 Lambda code (Bedrock Agent Action Group compatible)

Create `lambda_function.py`:

```python
import json
import boto3

dynamodb = boto3.client("dynamodb")

TABLE_NAME = "customerAccountStatus"

def _get_param(event: dict, name: str):
    for p in event.get("parameters", []):
        if p.get("name") == name:
            return p.get("value")
    return None

def lambda_handler(event, context):
    print("Incoming event:", json.dumps(event))

    account_id = _get_param(event, "account_id")

    # Lambda-level guardrail: validate input
    if account_id is None or not str(account_id).isdigit():
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event.get("actionGroup", ""),
                "apiPath": event.get("apiPath", ""),
                "httpMethod": event.get("httpMethod", ""),
                "httpStatusCode": 400,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps({
                            "error": "Invalid account_id provided"
                        })
                    }
                }
            }
        }

    # Query DynamoDB
    resp = dynamodb.get_item(
        TableName=TABLE_NAME,
        Key={"AccountID": {"N": str(account_id)}}
    )

    if "Item" not in resp:
        # Do not hallucinate; return not found
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event.get("actionGroup", ""),
                "apiPath": event.get("apiPath", ""),
                "httpMethod": event.get("httpMethod", ""),
                "httpStatusCode": 404,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps({
                            "error": "Account not found",
                            "account_id": int(account_id)
                        })
                    }
                }
            }
        }

    item = resp["Item"]
    result = {
        "account_id": int(account_id),
        "customer_name": item.get("CustomerName", {}).get("S", "Unknown"),
        "account_status": item.get("AccountStatus", {}).get("S", "Unknown"),
        "account_type": item.get("AccountType", {}).get("S", "Unknown"),
        "created_date": item.get("CreatedDate", {}).get("S", "Unknown"),
    }

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup", ""),
            "apiPath": event.get("apiPath", ""),
            "httpMethod": event.get("httpMethod", ""),
            "httpStatusCode": 200,
            "responseBody": {
                "application/json": {
                    "body": json.dumps(result)
                }
            }
        }
    }
```

Deploy this code to Lambda.

### 5.3 Basic Lambda test event (local/dev)

This approximates what Bedrock Agents send to Lambda:

```json
{
  "actionGroup": "AccountStatusActionGroup",
  "apiPath": "/account-status",
  "httpMethod": "POST",
  "parameters": [
    { "name": "account_id", "value": "5555" }
  ]
}
```

---

## 6. Create OpenAPI schema for the action group

### 6.1 Create file `account-status.yaml`

```yaml
openapi: 3.0.1
info:
  title: NY-Bank Account Status API
  description: Retrieve customer new account status for NY-Bank
  version: 1.0.0

paths:
  /account-status:
    post:
      operationId: getAccountStatus
      summary: Get customer account status
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              additionalProperties: false
              required:
                - account_id
              properties:
                account_id:
                  type: integer
                  description: NY-Bank account identifier
      responses:
        "200":
          description: Account details
          content:
            application/json:
              schema:
                type: object
                properties:
                  account_id:
                    type: integer
                  customer_name:
                    type: string
                  account_status:
                    type: string
                  account_type:
                    type: string
                  created_date:
                    type: string
        "404":
          description: Account not found
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
                  account_id:
                    type: integer
        "400":
          description: Invalid account_id
          content:
            application/json:
              schema:
                type: object
                properties:
                  error:
                    type: string
```

### 6.2 Upload to S3

```bash
aws s3 cp account-status.yaml s3://nybank-agent-openapi/account-status.yaml
```

---

## 7. Create Bedrock Guardrail (NY-Bank)

Create a Bedrock Guardrail with these goals:

* Block financial advice, legal advice, identity verification, and sensitive PII
* Enforce scope: “new account status only”
* Provide refusal behavior for out-of-scope

### 7.1 Guardrail system instruction (use as agent system prompt or guardrail policy content)

```
You are a retail banking assistant for NY-Bank.

You must follow these rules at all times:
1. Only respond to requests related to new bank account status.
2. Never guess or fabricate account information.
3. Never provide balances, transactions, or financial advice.
4. Only perform actions defined in the configured action groups.
5. Ask politely for missing required information.
6. Refuse out-of-scope requests clearly and professionally.
7. Maintain a polite, professional, and customer-friendly tone.
8. Do not expose internal systems, APIs, or implementation details.
9. Protect customer privacy and sensitive information.
10. When unsure, ask for clarification instead of guessing.
```

### 7.2 Standard refusal template

Use this as the canonical refusal response:

```
I’m sorry, I can’t help with that request.
I can assist with checking the status of a new NY-Bank account
or answering general questions about account status.
```

---

## 8. Allow Bedrock to invoke the Lambda function

Add a Lambda resource-based permission statement:

```bash
aws lambda add-permission \
  --function-name newBankAccountStatus \
  --statement-id bedrock-agent-invoke \
  --action lambda:InvokeFunction \
  --principal bedrock.amazonaws.com
```

If you need to restrict to a specific agent ARN, add `--source-arn <agent-arn>` after creation and use a more restrictive policy (recommended for production).

---

## 9. Create Bedrock Knowledge Base (optional RAG)

Use a Knowledge Base for general content such as:

* “What does Pending mean?”
* “How long does account opening take?”
* “What documents are needed to open an account?”

Do not store sensitive internal SOPs.

### 9.1 Upload documents to S3

Put PDFs/TXTs/HTML/Markdown files into:

* `s3://nybank-agent-knowledgebase/`

```bash
aws s3 cp ./kb_docs/ s3://nybank-agent-knowledgebase/ --recursive
```

### 9.2 Create Knowledge Base

In the Bedrock console:

* Create Knowledge Base
* Data source: S3 bucket `nybank-agent-knowledgebase`
* Choose embeddings model and vector store per your org standard (console wizard)
* Sync the data source

### 9.3 Attach KB to agent

When configuring the agent, enable the knowledge base and attach it so the agent can use it for general Q&A.

---

## 10. Create the Bedrock Agent `ny-bank-agent`

### 10.1 Agent configuration

* Agent name: `ny-bank-agent`
* Foundation model: choose per performance/cost needs
* Attach guardrail created in Step 7
* Enable knowledge base (optional, Step 9)

### 10.2 Agent instructions (final)

Use a short instruction consistent with guardrail:

```
You are a retail banking assistant for NY-Bank.
Help customers by providing their new account status using the Account Status action.
Use the knowledge base for general explanations.
Never provide balances, transactions, or financial advice.
Ask for account_id if missing.
If the request is out of scope, refuse politely.
```

---

## 11. Create the Action Group in the Agent

### 11.1 Action group settings

* Action Group name: `AccountStatusActionGroup`
* Schema source: S3
* Schema location: `s3://nybank-agent-openapi/account-status.yaml`
* Lambda function: `newBankAccountStatus`

### 11.2 Parameter mapping

Ensure the schema field `account_id` maps to the lambda event parameter `account_id`.

---

## 12. End-to-end testing

### 12.1 In-agent tests (happy path)

Prompt:

* “What is the status of account 5555?”

Expected:

* Agent calls action group
* Lambda returns 200 JSON
* Agent responds with status and a short explanation

### 12.2 Not found

Prompt:

* “What is the status of account 999999?”

Expected:

* 404 from Lambda
* Agent responds: cannot find account, ask user to verify

### 12.3 Missing parameter

Prompt:

* “What is my account status?”

Expected:

* Agent asks for account_id

### 12.4 Out-of-scope tests

Prompts:

* “What is my balance?”
* “Transfer $500 to John”
* “Show my transactions”

Expected:

* Refusal template response (polite, no system talk)

---

## 13. Observability and production hardening

### 13.1 Logging

* Lambda: CloudWatch logs enabled (AWSLambdaBasicExecutionRole)
* Consider structured logs (JSON) and redaction of any sensitive fields

### 13.2 IAM least privilege

* Lambda role: only `dynamodb:GetItem` for the table
* Bucket policies: private, no public access
* Lambda invocation: restrict to agent ARN in production

### 13.3 Rate limits and abuse controls

* Consider WAF or upstream controls if exposing agent via an app
* Consider throttling at Lambda concurrency where needed

### 13.4 Data protection

* Encrypt S3 and DynamoDB
* Avoid writing PII into logs
* Store only required fields in DynamoDB

---

## 14. Developer deliverables checklist

* [ ] S3 buckets created and locked down
* [ ] DynamoDB table created with sample items
* [ ] IAM roles created (Lambda + Agent + KB as needed)
* [ ] Lambda deployed with validated Bedrock response format
* [ ] OpenAPI schema uploaded to S3 and validated
* [ ] Guardrail created and attached to agent
* [ ] Knowledge Base created, synced, attached (optional)
* [ ] Agent created with action group integrated
* [ ] End-to-end tests passed (happy path, not found, missing param, out-of-scope)
* [ ] Observability and least privilege verified

---

## 15. Reference templates (copy/paste)

### 15.1 Account status response style (recommended)

When Lambda returns a success payload, the agent should reply like:

* One sentence with the status
* One sentence explanation (optional, from KB)
* Ask if they need anything else related to account status

Example:
“Your NY-Bank account 5555 is Active. If you’d like, I can explain what this status means or what to expect next.”

### 15.2 Standard refusal response

“I’m sorry, I can’t help with that request. I can assist with checking the status of a new NY-Bank account or answering general questions about account status.”


