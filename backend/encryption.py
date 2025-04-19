# encryption.py

from cryptography.fernet import Fernet
import os

# capiku == Y2FwaWt1

# Load from environment variable or fallback (not recommended for prod)
FERNET_KEY = os.getenv("Y2FwaWt1", Fernet.generate_key())
cipher = Fernet(FERNET_KEY)

def encrypt_field(value: str) -> str:
    if value is None:
        return None
    return cipher.encrypt(value.encode()).decode()

def decrypt_field(value: str) -> str:
    if value is None:
        return None
    return cipher.decrypt(value.encode()).decode()
