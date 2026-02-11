Below are **least-privilege IAM policies** you can hand to the dev team for **NY-Bank** covering:

* DynamoDB read (for Lambda)
* Bedrock Agent service role (invoke Lambda + invoke Nova/Claude models + retrieve from Knowledge Base)
* Knowledge Base document access (S3 “bank policies” docs)
* Trust policies (who can assume the roles)

Replace all placeholders like `<REGION>`, `<ACCOUNT_ID>`, `<KB_ID>`, `<FUNCTION_NAME>`, `<DOCS_BUCKET>`.

---

# 1) Lambda execution role (DynamoDB read + logs)

## 1.1 Trust policy (Lambda assumes the role)

**Role name:** `nybank-lambda-exec-role`
**Trust policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaAssumeRole",
      "Effect": "Allow",
      "Principal": { "Service": "lambda.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## 1.2 Permissions policy (least privilege)

Attach **AWS managed** policy:

* `AWSLambdaBasicExecutionRole`

Attach this **inline policy** for DynamoDB:

```json
{
  "Version": "2012-10-
  "Statement": [
    {
      "Sid": "DynamoDBGetItemOnly",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem"
      ],
      "Resource": "arn:aws:dynamodb:<REGION>:<ACCOUNT_ID>:table/customerAccountStatus"
    }
  ]
}
```

---

# 2) Bedrock Agent service role (invoke Lambda + models + Knowledge Base retrieval)

This is the role **Bedrock Agents** uses to call your tools and KB.

## 2.1 Trust policy (Bedrock assumes the role)

**Role name:** `nybank-bedrock-agent-role`
**Trust policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockAssumeRoleForAgents",
      "Effect": "Allow",
      "Principal": { "Service": "bedrock.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## 2.2 Permissions policy (least privilege)

### A) Invoke the Action Group Lambda

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeAccountStatusLambda",
      "Effect": "Allow",
      "Action": [
        "lambda:InvokeFunction"
      ],
      "Resource": [
        "arn:aws:lambda:<REGION>:<ACCOUNT_ID>:function:newBankAccountStatus",
        "arn:aws:lambda:<REGION>:<ACCOUNT_ID>:function:newBankAccountStatus:*"
      ]
    }
  ]
}
```

### B) Allow the agent to invoke only the approved foundation models (Nova + Claude)

**Important notes**

* Bedrock model ARNs differ by region/account and whether they’re “foundation models” or “inference profiles”.
* If your org uses **Inference Profiles**, you should restrict to profile ARNs instead.
* If you don’t have exact ARNs yet, start with `Resource: "*"` + `Condition` on `bedrock:ModelId`, then tighten.

**Recommended (tight) using `bedrock:ModelId` condition:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeOnlyApprovedModels",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*",
      "Condition": {
        "ForAnyValue:StringEquals": {
          "bedroc[
            "amazon.nova-pro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-micro-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0"
          ]
        }
      }
    }
  ]
}
```

If you use different Claude/Nova versions, replace the model IDs accordingly.

### C) Knowledge Base runtime permissions (retrieve only from your KB)

If your agent uses **Knowledge Base** (Retrieve / RetrieveAndGenerate), grant access to the specific KB:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RetrieveFromNYBankKnowledgeBase",
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate"
      ],
      "Resource": "arn:aws:bedrock:<REGION>:<ACCOUNT_ID>:knowledge-base/<KB_ID>"
    }
  ]
}
```

---

# 3) Knowledge Base execution role (S3 docs access for “bank policies”)

Depending on how you created the Knowledge Base, Bedrock may ask for a role to access yS3 docs. Use this role to let the KB ingestion pipeline read documents.

## 3.1 Trust policy (Bedrock assumes the role for KB ingestion)

**Role name:** `nybank-bedrock-kb-role`
**Trust policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockAssumeRoleForKB",
      "Effect": "Allow",
      "Principal": { "Service": "bedrock.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

## 3.2 Permissions policy (read-only on docs bucket/prefix)

Assume:

* Bucket: `<DOCS_BUCKET>` (example: `nybank-agent-knowledgebase`)
* Prefix: `policies/` (recommended to isolate only bank policy docs)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListDocsBucket",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::<DOCS_BUCKET>",
      "Condition": {
        "StringLike": {
          "s3:prefix": [
            "policies/*"
          ]
        }
      }
    },
    {
      "Sid": "ReadDocsObjects",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::<DOCS_BUCKET>/policies/*"
    }
  ]
}
```

---

# 4) Lambda resource-based policy (restrict invocation to your Agent)

Even with role permissions, you should restrict Lambda invocation at the function level.

## 4.1 Allow Bedrock to invoke Lambda (basic)

```bash
aws lambda add-permission \
  --function-name newBankAccountStatus \
  --statement-id allow-bedrock-agent \
  --action lambda:InvokeFunction \
  --principal bedrock.amazonaws.com
```

## 4.2 Recommended: restrict to a specific agent ARN (tightest)

After you create the agent, get its ARN and apply:

```bash
aws lambda add-permission \
  --function-name newBankAccountStatus \
  --statement-id allow-specific-bedrock-agent \
  --action lambda:InvokeFunction \
  --principal bedrock.amazonaws.com \
  --source-arn arn:aws:bedrock:<REGION>:<ACCOUNT_ID>:agent/<AGENT_ID>
```

---

# 5) Optional: S3 bucket policy to enforce least privilege (docs bucket)

If your security team prefers bucket policies, you can restrict S3 access to only the KB role.

**Bucket policy on `<DOCS_BUCKET>`** (example; adjust role ARN):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyNonTLS",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::<DOCS_BUCKET>",
        "arn:aws:s3:::<DOCS_BUCKET>/*"
      ],
      "Condition": {
        "Bool": { "aws:SecureTransport": "false" }
      }
    },
    {
      "Sid": "AllowKBRoleReadPoliciesOnly",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::<ACCOUNT_ID>:role/nybank-bedrock-kb-role"
      },
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::<DOCS_BUCKET>",
      "Condition": {
        "StringLike": { "s3:prefix": [ "policies/*" ] }
      }
    },
    {
      "Sid": "AllowKBRoleGetPoliciesOnly",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::<ACCOUNT_ID>:role/nybank-bedrock-kb-role"
      },
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::<DOCS_BUCKET>/policies/*"
    }
  ]
}
```

---

# 6) What you still might need (vector store permissions)

Knowledge Bases typically require a vector store (commonly OpenSearch Serverless). Least-privilege for that depends on your choice:

* OpenSearch Serverless (`aoss:*` scoped to collection/index)
* Aurora/RDS (if using that backend)
* KMS permissions if you choose SSE-KMS


