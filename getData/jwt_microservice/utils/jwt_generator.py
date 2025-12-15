import jwt
import time

# NOTE: The 'secret' here MUST match the secret key used by your Wix backend
# to validate the JWT. This is usually stored securely as an environment variable.

ALGORITHM = "HS256"


def generate_jwt(
    secret: str, ttl_seconds: int = 900, subject: str = "fastapi-client"
) -> str:
    """
    Generates a signed JWT token.

    Args:
        secret (str): The secret key used for signing (HS256).
        ttl_seconds (int): Time to live for the token in seconds (default 15 minutes).
        subject (str): The subject claim (sub) for the token.

    Returns:
        str: The generated JWT token.
    """
    now = int(time.time())

    # Standard JWT claims: subject, issued at, expiration
    payload = {"sub": subject, "iat": now, "exp": now + ttl_seconds}

    # Encode and sign the token
    return jwt.encode(payload, secret, algorithm=ALGORITHM)
