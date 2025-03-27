# smolavision/batch/utils.py
import logging
from typing import List

logger = logging.getLogger(__name__)

def calculate_batch_size_mb(batch: List[str]) -> float:
    """
    Calculate the size of a batch of image data in megabytes.

    Args:
        batch: List of base64 encoded image data strings.

    Returns:
        The size of the batch in megabytes.
    """
    total_bytes = sum(len(image_data) for image_data in batch)
    return total_bytes / (1024 * 1024)