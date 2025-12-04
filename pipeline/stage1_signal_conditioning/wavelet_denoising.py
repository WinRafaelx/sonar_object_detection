"""
2D Discrete Wavelet Transform (DWT) Denoising

Purpose: Remove speckle noise while preserving sharp edges of target shadows
"""
import numpy as np
import pywt
from typing import Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)


class WaveletDenoiser:
    """
    Implements 2D DWT denoising for sonar images
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        mode: str = 'soft',
        threshold_multiplier: float = 0.5
    ):
        """
        Initialize wavelet denoiser
        
        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'bior2.2')
            mode: Thresholding mode ('soft' or 'hard')
            threshold_multiplier: Multiplier for automatic threshold calculation
        """
        self.wavelet = wavelet
        self.mode = mode
        self.threshold_multiplier = threshold_multiplier
        logger.info(f"WaveletDenoiser initialized: wavelet={wavelet}, mode={mode}")
    
    def denoise(self, image: np.ndarray, level: Optional[int] = None) -> np.ndarray:
        """
        Apply 2D DWT denoising to sonar image
        
        Args:
            image: 2D sonar image (normalized to [0, 1])
            level: Decomposition level (auto if None)
            
        Returns:
            denoised_image: Denoised image
        """
        if level is None:
            # Auto-determine level based on image size
            level = pywt.dwt_max_level(min(image.shape), self.wavelet)
            level = min(level, 4)  # Cap at 4 levels
        
        logger.debug(f"Applying DWT denoising: level={level}, wavelet={self.wavelet}")
        
        # Decompose image into sub-bands
        coeffs = pywt.wavedec2(image, self.wavelet, level=level)
        
        # Calculate threshold using universal threshold rule
        # Estimate noise from finest detail coefficients
        detail_coeffs = coeffs[0]  # Approximation coefficients
        if len(coeffs) > 1:
            # Use horizontal detail coefficients for noise estimation
            detail_coeffs = coeffs[1][0]  # Horizontal detail
        
        # Median absolute deviation (MAD) for robust noise estimation
        sigma = np.median(np.abs(detail_coeffs - np.median(detail_coeffs))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(image.size)) * self.threshold_multiplier
        
        # Apply thresholding to detail coefficients (preserve approximation)
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        
        for i in range(1, len(coeffs)):
            # Threshold each detail sub-band
            thresh_coeffs = tuple(
                pywt.threshold(detail, threshold, mode=self.mode)
                for detail in coeffs[i]
            )
            coeffs_thresh.append(thresh_coeffs)
        
        # Reconstruct image
        denoised = pywt.waverec2(coeffs_thresh, self.wavelet)
        
        # Crop to original size (wavelet transform may add padding)
        denoised = denoised[:image.shape[0], :image.shape[1]]
        
        # Ensure values stay in valid range
        denoised = np.clip(denoised, 0, 1)
        
        logger.debug(f"Denoising complete. Threshold: {threshold:.4f}")
        
        return denoised

