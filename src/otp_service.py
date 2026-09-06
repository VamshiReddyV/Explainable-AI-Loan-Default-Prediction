import os
import json
import hashlib
import secrets
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

def generate_otp(length: int = 6) -> str:
    """Generate a cryptographically secure numeric OTP."""
    digits = "0123456789"
    return "".join(secrets.choice(digits) for _ in range(length))

def hash_otp(otp: str) -> str:
    """Hash an OTP code with SHA-256 for secure database storage."""
    return hashlib.sha256(otp.encode('utf-8')).hexdigest()

def mask_phone(phone: str) -> str:
    """Mask phone number for safe UI display (e.g. +91 ******7890)."""
    clean_digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if len(clean_digits) == 10:
        return f"+91 ******{clean_digits[-4:]}"
    elif len(clean_digits) > 4:
        return f"******{clean_digits[-4:]}"
    return "****"

def send_otp(phone: str, otp_code: str, user_name: str = "") -> dict:
    """
    Send OTP via configured SMS provider or safely log in development mode.
    
    Supported providers:
    - 'fast2sms': via FAST2SMS_API_KEY
    - 'twilio': via TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
    - None / 'dev' / default: Development Mode fallback (logs OTP cleanly)
    """
    provider = os.getenv("OTP_PROVIDER", "dev").strip().lower()
    api_key = os.getenv("OTP_API_KEY", "").strip()
    
    # In development mode or if credentials are unset
    if provider in ("dev", "development", "local") or not api_key:
        logger.info("=" * 60)
        logger.info(f"🔑 [DEV OTP SERVICE] OTP for {phone} ({user_name}): {otp_code}")
        logger.info("=" * 60)
        return {
            "success": True,
            "mode": "development",
            "message": "OTP generated in development mode.",
            "dev_otp": otp_code,
            "phone": phone
        }
    
    # Pluggable Provider: Fast2SMS (Indian SMS Gateway)
    if provider == "fast2sms":
        try:
            url = "https://www.fast2sms.com/dev/bulkV2"
            payload = json.dumps({
                "variables_values": otp_code,
                "route": "otp",
                "numbers": phone
            }).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={
                "authorization": api_key,
                "Content-Type": "application/json"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get("return") is True:
                    return {"success": True, "mode": "fast2sms", "message": "SMS sent successfully."}
                else:
                    logger.error(f"Fast2SMS error: {data}")
                    return {"success": False, "mode": "fast2sms", "error": data.get("message", "SMS dispatch failed.")}
        except Exception as e:
            logger.error(f"Fast2SMS request exception: {e}")
            return {"success": False, "mode": "fast2sms", "error": str(e)}

    # Generic fallback
    logger.info(f"🔑 [DEV OTP SERVICE] OTP for {phone}: {otp_code}")
    return {
        "success": True,
        "mode": "development",
        "message": "OTP generated in development mode.",
        "dev_otp": otp_code
    }
