import numpy as np
import os
import pyexr
import torch

# PU21 parameters
_A = 0.001908
_B = 0.0078
_L_MIN = 0.005
_L_MAX = 10000.0
_L_MIN_LOG2 = np.log2(_L_MIN)

def pu21_encode(L):
    # Clamp to PU21 valid range and encode
    Lc = np.clip(L, _L_MIN, _L_MAX)
    x = np.log2(Lc) - _L_MIN_LOG2
    return _A * (x * x) + _B * x  # in [0,1] for L in [_L_MIN, _L_MAX]

def pu21_encode_rgb(rgb_abs):
    return pu21_encode(rgb_abs)

def pu21_decode(V):
    """
    PU21 inverse function: converts from [0,1] range back to original luminance
    Formula: L = 2^((2*a*L_min - b + sqrt(b^2 + 4*a*V)) / (2*a))
    where a = _A, b = _B, L_min = log2(_L_MIN)
    """
    # Ensure V is in valid range [0,1]
    V = np.clip(V, 0.0, 1.0)
    
    # Apply the inverse formula
    # v_PU21^(-1)(V) = 2^((2*a*L_min - b + sqrt(b^2 + 4*a*V)) / (2*a))
    a = _A
    b = _B
    L_min_log2 = _L_MIN_LOG2
    
    # Calculate discriminant: b^2 + 4*a*V
    discriminant = b * b + 4 * a * V
    
    # Calculate the exponent: (2*a*L_min - b + sqrt(discriminant)) / (2*a)
    exponent = (2 * a * L_min_log2 - b + np.sqrt(discriminant)) / (2 * a)
    
    # Calculate L = 2^exponent
    L = np.power(2.0, exponent)
    
    # Clamp to valid range
    L = np.clip(L, _L_MIN, _L_MAX)
    
    return L

def pu21_decode_rgb(V_rgb):
    return pu21_decode(V_rgb)

def scale_to_L_peak(rgb, L_peak=4000.0):
    # m = L_peak / max(I(:))  (MATLAB strategy)
    max_val = float(np.max(rgb))
    if max_val <= 0 or not np.isfinite(max_val):
        return np.full_like(rgb, _L_MIN, dtype=np.float32), 1.0
    m = L_peak / max_val
    rgb_abs = (rgb * m).astype(np.float32)
    return rgb_abs, m

def read_hdr_image(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ['.hdr', '.exr']:
        hdr_image = pyexr.read(filepath)
        # pyexr returns (H, W, C) in RGB order
        if len(hdr_image.shape) == 3 and hdr_image.shape[2] >= 3:
            # Ensure we have 3 channels (RGB)
            hdr_image = hdr_image[:, :, :3]
        elif len(hdr_image.shape) == 2:
            # Single channel, convert to RGB
                hdr_image = np.stack([hdr_image, hdr_image, hdr_image], axis=2)
        hdr_image = np.maximum(hdr_image, 0).astype(np.float32)
        return hdr_image
    else:
        print(f"Unsupported HDR format: {filepath}. Supported formats: .hdr, .exr")
        return None

def save_hdr_image(hdr_image, output_path):
    if hdr_image.dtype != np.float32:
        hdr_image = hdr_image.astype(np.float32)
    pyexr.write(output_path, hdr_image)

def recover_hdr_from_pu21(pu21_image, exposure_multiplier):
    """
    Recover HDR image from PU21 encoded image
    Args:
        pu21_image: PU21 encoded image in [0,1] range
        exposure_multiplier: The multiplier used during encoding (m from scale_to_L_peak)
    Returns:
        HDR image in original luminance range
    """
    # Decode from PU21 to absolute luminance
    hdr_abs = pu21_decode_rgb(pu21_image)
    
    # Recover original HDR by dividing by exposure multiplier
    hdr_original = hdr_abs / exposure_multiplier
    
    return hdr_original

def compute_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

def get_luminance_percentile(hdr_image, percentile=99.5):
    luminance = compute_luminance(hdr_image)
    return np.percentile(luminance, percentile)
