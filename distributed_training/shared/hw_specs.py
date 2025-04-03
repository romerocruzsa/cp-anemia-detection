import platform
import random
import socket

def get_device_specs():
    return {
        "ram": round(random.uniform(1024, 8192), 2),  # in MB
        "gpu": random.random() > 0.4,
        "cpu": platform.processor(),
        "device": platform.node(),
        "ip": socket.gethostbyname(socket.gethostname())
    }

def classify_node(specs: dict) -> str:
    """
    Classify device as 'trainer', 'inference', or 'dashboard'.
    """
    ram = specs.get("ram", 0)
    has_gpu = specs.get("gpu", False)

    if ram >= 4096 and has_gpu:
        return "trainer"
    elif ram >= 512:
        return "inference"
    else:
        return "dashboard"

def generate_tags(specs: dict) -> list:
    tags = []
    if specs.get("gpu"):
        tags.append("gpu")
    if specs.get("ram", 0) >= 4096:
        tags.append("high-mem")
    elif specs.get("ram", 0) < 1024:
        tags.append("low-resource")
    tags.append("auto")
    return tags
