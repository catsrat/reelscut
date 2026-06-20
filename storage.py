"""
Optional Cloudflare R2 storage for finished clips.

If these env vars are set, rendered clips are uploaded to R2 and served from
there (so they survive container restarts). If not set, the app falls back to
serving clips from local disk — nothing breaks either way.

  R2_ACCOUNT_ID         your Cloudflare account id
  R2_ACCESS_KEY_ID      R2 API token access key
  R2_SECRET_ACCESS_KEY  R2 API token secret
  R2_BUCKET             bucket name (e.g. "reelscut")
"""

import os

_LINK_TTL = 7 * 24 * 3600  # presigned URL valid for 7 days


def enabled():
    return all(os.environ.get(k) for k in
               ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def _client():
    if not enabled():
        return None
    import boto3  # lazy import so the app runs without boto3 when R2 is off
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_and_url(local_path, key):
    """Upload a file to R2 and return a presigned URL, or None if R2 is off/fails."""
    client = _client()
    if not client:
        return None
    bucket = os.environ["R2_BUCKET"]
    try:
        client.upload_file(
            local_path, bucket, key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        return client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key},
            ExpiresIn=_LINK_TTL,
        )
    except Exception:
        return None  # never let a storage hiccup break the job
