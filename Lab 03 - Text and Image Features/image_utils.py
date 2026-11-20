"""
Lab 03: Text and Image Feature Engineering Utilities
Course: Introduction to Artificial Intelligence (AI-101L)
Instructor: Dr. Muhammad Awais

This module provides reusable utilities for converting unstructured image data
into normalized, machine-understandable 1D numerical feature vectors.
"""

import os
import numpy as np
from PIL import Image


def image_to_features(image_input, size=(64, 64), normalize=True):
    """
    Transform an image (from file path or PIL Image object) into a standardized,
    grayscale, flattened 1D feature vector.

    Parameters
    ----------
    image_input : str or PIL.Image.Image or np.ndarray
        File path to an image, a PIL Image instance, or a numpy array.
    size : tuple of int, default=(64, 64)
        Target dimensions (width, height) for image downsampling/resizing.
    normalize : bool, default=True
        If True, scales pixel intensity values from [0, 255] to [0.0, 1.0].

    Returns
    -------
    np.ndarray
        1D numpy array of shape (size[0] * size[1],) representing the feature vector.
    """
    # 1. Load image if string path is provided
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found at: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # 2. Convert to Grayscale ('L' mode: luminance)
    if img.mode != 'L':
        img = img.convert('L')

    # 3. Resize to standardized fixed dimensions
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)

    # 4. Convert to numerical numpy array
    pixel_array = np.array(img, dtype=np.float32)

    # 5. Normalization [0, 255] -> [0.0, 1.0]
    if normalize:
        pixel_array /= 255.0

    # 6. Flatten 2D matrix into 1D feature vector
    feature_vector = pixel_array.flatten()
    return feature_vector


def batch_images_to_matrix(image_list, size=(64, 64), normalize=True):
    """
    Process a collection of images and stack their feature vectors into a
    2D design matrix X of shape (num_samples, size[0] * size[1]).

    Parameters
    ----------
    image_list : list of (str or PIL.Image.Image or np.ndarray)
        List of images to process.
    size : tuple of int, default=(64, 64)
        Target dimensions for resizing.
    normalize : bool, default=True
        Whether to normalize pixel values to [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Design matrix X of shape (len(image_list), size[0] * size[1]).
    """
    feature_vectors = [image_to_features(img, size=size, normalize=normalize) for img in image_list]
    return np.vstack(feature_vectors)


if __name__ == "__main__":
    print("Testing image_to_features utility...")
    # Create synthetic test pattern
    test_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    vec = image_to_features(test_img, size=(64, 64), normalize=True)
    print(f"Input shape: (128, 128, 3)")
    print(f"Output vector shape: {vec.shape} (Expected: (4096,))")
    print(f"Value range: min={vec.min():.3f}, max={vec.max():.3f}")
    assert vec.shape == (4096,), "Dimension mismatch in feature flattening"
    assert 0.0 <= vec.min() and vec.max() <= 1.0, "Normalization out of bounds"
    print("All image_utils tests passed successfully!")
