"""
Visualization utilities for debugging pipeline stages
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


def visualize_stage1_output(
    original: np.ndarray,
    processed: np.ndarray,
    bottom_line: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Stage 1: Signal Conditioning"
):
    """
    Visualize Stage 1 preprocessing results
    
    Args:
        original: Original sonar image
        processed: Processed sonar image
        bottom_line: Seabed line from Viterbi (optional)
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(original, cmap='gray', aspect='auto')
    axes[0].set_title('Original Sonar Image')
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap='gray', aspect='auto')
    axes[1].set_title('Processed Image')
    if bottom_line is not None:
        axes[1].plot(bottom_line, 'r-', linewidth=2, label='Seabed Line')
        axes[1].legend()
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_detections(
    image: np.ndarray,
    detections: list,
    save_path: Optional[Path] = None,
    title: str = "Detection Results"
):
    """
    Visualize detection results on sonar image
    
    Args:
        image: Sonar image
        detections: List of detections with format [x1, y1, x2, y2, conf, cls]
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image, cmap='gray', aspect='auto')
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            fill=False, edgecolor='red', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1-5, f'{cls}: {conf:.2f}', 
                color='red', fontsize=10, fontweight='bold')
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

