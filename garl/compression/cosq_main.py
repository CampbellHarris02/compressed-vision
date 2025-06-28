import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import json
import random
from random import randrange

from cosq_extr import (
    grey_image_array, crop_to_square, 
    compress_image_with_codebook, implement_channel, 
    dequantize_image, create_montage,
    compress_image_with_lloydmax_codebook, dequantize_image_lloydmax
)
import image_formatting as imf


def main():
    # Load original COSQ codebooks
    cosq_codebook_path = 'cosq_codebooks.json'
    with open(cosq_codebook_path, 'r') as f:
        codebooks_dict = json.load(f)
    
    # Load Lloyd–Max codebooks
    lm_codebook_path = 'lloydmax_codebooks.json'
    with open(lm_codebook_path, 'r') as f:
        LM_codebooks_dict = json.load(f)

    # Prepare the input image
    image_path = 'images/goldhill.jpg'
    raw_image = Image.open(image_path)
    grey_image = grey_image_array(image_path)
    image_array = crop_to_square(grey_image)
    h, w = image_array.shape

    # Set parameters
    block_size = 8
    B_bits = 76  # (Possible options: 24, 58, or 76)
    noise_idx = 7

    # 1) Compress with COSQ codebook
    compressed_img_array, dcMean, dcStd, acStd = compress_image_with_codebook(
        image_array, codebooks_dict, B_bits, noise_idx, block_size
    )
    
    # 1b) Compress with Lloyd–Max codebook
    LM_compressed_img_array, dcMean_lm, dcStd_lm, acStd_lm = compress_image_with_lloydmax_codebook(
        image_array, LM_codebooks_dict, B_bits, block_size
    )

    # 2) Polya-based channel for both bitstreams
    R=1
    G=0
    B=99
    Del = 10
    M = 1
    channel_imposed_img = implement_channel(compressed_img_array, R, G, B, M, Del)
    LM_channel_imposed_img = implement_channel(LM_compressed_img_array, R, G, B, M, Del)

    # 3) Dequantize
    decoded_blocks = dequantize_image(
        channel_imposed_img, codebooks_dict, B_bits, noise_idx,
        block_size, dcMean, dcStd, acStd
    )
    LM_decoded_blocks = dequantize_image_lloydmax(
        LM_channel_imposed_img, LM_codebooks_dict, B_bits,
        block_size, dcMean_lm, dcStd_lm, acStd_lm
    )

    # 4) Reconstruct (montage)
    reconstructed_image = create_montage(decoded_blocks, h, w, block_size)
    LM_reconstructed_image = create_montage(LM_decoded_blocks, h, w, block_size)

    # --- Calculate PSNR for COSQ Reconstruction ---
    mse_cosq = np.mean((image_array - reconstructed_image)**2)
    if mse_cosq == 0:
        psnr_cosq = float('inf')
    else:
        psnr_cosq = 10 * np.log10((255**2) / mse_cosq)
    print(f"COSQ PSNR: {psnr_cosq:.2f} dB")

    # --- Calculate PSNR for Lloyd–Max Reconstruction ---
    mse_lm = np.mean((image_array - LM_reconstructed_image)**2)
    if mse_lm == 0:
        psnr_lm = float('inf')
    else:
        psnr_lm = 10 * np.log10((255**2) / mse_lm)
    print(f"Lloyd–Max PSNR: {psnr_lm:.2f} dB")

    # --- Plot 3 Images in one figure ---
    plt.figure(figsize=(14, 6))

    # Original
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    # COSQ Reconstructed
    plt.subplot(1, 3, 2)
    plt.title(f"COSQ Reconstructed\nPSNR={psnr_cosq:.2f} dB")
    plt.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    # Lloyd–Max Reconstructed
    plt.subplot(1, 3, 3)
    plt.title(f"Lloyd–Max Reconstructed\nPSNR={psnr_lm:.2f} dB")
    plt.imshow(LM_reconstructed_image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
