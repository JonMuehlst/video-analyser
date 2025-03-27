# smolavision/batch/creator.py
import logging
import base64
from typing import List, Dict, Any
from smolavision.video.types import Frame
from smolavision.batch.types import Batch
from smolavision.batch.utils import calculate_batch_size_mb  # Import utils

logger = logging.getLogger(__name__)

def create_batches(
    frames: List[Frame],
    max_batch_size_mb: float = 10.0,
    max_images_per_batch: int = 15,
    overlap_frames: int = 2
) -> List[Batch]:
    """
    Create batches of frames for efficient analysis.

    Args:
        frames: List of Frame objects
        max_batch_size_mb: Maximum size of a batch in megabytes (image data).
        max_images_per_batch: Maximum number of images in a batch.
        overlap_frames: Number of overlapping frames between batches.

    Returns:
        List of Batch objects.
    """
    logger.info(f"Creating batches (max size: {max_batch_size_mb} MB, max images: {max_images_per_batch}, overlap: {overlap_frames})")

    batches: List[Batch] = []
    current_batch_frames: List[int] = []
    current_batch_image_data: List[str] = []
    current_batch_timestamps: List[float] = []
    current_batch_ocr_text: List[str] = []
    
    # Track actual image bytes
    current_batch_size_bytes = 0

    for i, frame in enumerate(frames):
        frame_size_bytes = len(frame.image_data) * 3 / 4  # approx base64 to image size

        # Create hypothetical new batch to test limit size. This code path is important to avoid exceeding the limits.
        hypothetical_frames = current_batch_frames + [frame.frame_number]
        hypothetical_image_data = current_batch_image_data + [frame.image_data]
        hypothetical_timestamps = current_batch_timestamps + [frame.timestamp]
        hypothetical_ocr_text = current_batch_ocr_text + [frame.ocr_text or ""]
        
        #Use the helper to calculate new batch size
        new_batch_size_bytes = current_batch_size_bytes + frame_size_bytes

        #Check the size with helper and the max_images_per_batch length
        if len(hypothetical_frames) > max_images_per_batch or \
            (new_batch_size_bytes) > (max_batch_size_mb * 1024 * 1024):

            # Create a new batch with existing frames
            batches.append(
                Batch(
                    frames=current_batch_frames,
                    image_data=current_batch_image_data,
                    timestamps=current_batch_timestamps,
                    ocr_text=current_batch_ocr_text #OCR data
                )
            )

            # Start a new batch, including overlapping frames if applicable
            current_batch_frames = []
            current_batch_image_data = []
            current_batch_timestamps = []
            current_batch_ocr_text = []
            current_batch_size_bytes = 0

            # Add overlapping frames from the previous batch, if any
            for j in range(max(0, i - overlap_frames), i): #overlap_frames value is also taken into account
                overlap_frame = frames[j]
                current_batch_frames.append(overlap_frame.frame_number)
                current_batch_image_data.append(overlap_frame.image_data)
                current_batch_timestamps.append(overlap_frame.timestamp)
                current_batch_ocr_text.append(overlap_frame.ocr_text or "")
                #Update with current size calculation
                current_batch_size_bytes += len(overlap_frame.image_data) * 3 / 4

        # Add the current frame to the batch
        current_batch_frames.append(frame.frame_number)
        current_batch_image_data.append(frame.image_data)
        current_batch_timestamps.append(frame.timestamp)
        current_batch_ocr_text.append(frame.ocr_text or "")
        current_batch_size_bytes += frame_size_bytes

    # Create the last batch if there are any remaining frames
    if current_batch_frames:
        batches.append(
            Batch(
                frames=current_batch_frames,
                image_data=current_batch_image_data,
                timestamps=current_batch_timestamps,
                ocr_text = current_batch_ocr_text
            )
        )

    logger.info(f"Created {len(batches)} batches.")
    return batches