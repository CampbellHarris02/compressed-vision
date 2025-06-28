from PIL import Image
import numpy as np

def grey_image_array(image_path):
  """
    import the image and store in a variable
    return the image pixel value array
  """
  raw_image = Image.open(image_path)
  grey_image = raw_image.convert("L")
  image_array = np.array(grey_image)
  return image_array


def crop_to_square(image):
  height, width = image.shape[:2]
  min_side = min(height, width)
  start_x = (width - min_side) // 2
  start_y = (height - min_side) // 2
  cropped_image = image[start_y:start_y + min_side, start_x:start_x + min_side]
  return cropped_image

def blockify_image(image, block_size):
    """ Break the image into blocks of size block_size x block_size,
    handling edge cases where the image size is not divisible by block_size """
    blocks = []
    h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # Extract block, ensuring it has the correct dimensions
            block = image[i:min(i + block_size, h), j:min(j + block_size, w)]

            # Pad the block if it's smaller than block_size x block_size
            if block.shape != (block_size, block_size):
                pad_height = block_size - block.shape[0]
                pad_width = block_size - block.shape[1]
                block = np.pad(block, ((0, pad_height), (0, pad_width)), 'edge')

            blocks.append(block)
    return blocks

def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)  # Scale to 0-1 range

def create_montage(blocks, h, w, block_size):
    montage_image = np.zeros((h, w))
    block_idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = blocks[block_idx]
            block_height = min(block_size, h - i)
            block_width = min(block_size, w - j)
            montage_image[i:i+block_height, j:j+block_width] = block[:block_height, :block_width]
            block_idx += 1
    return montage_image


def encode_get_gaussian_coeff_vals(dctBlocks):
    gaussianVals = []
    for block in dctBlocks:
        # The DC coefficient is the first element in the 1D array
        gaussianVals.append(block[0])
    return gaussianVals


def encode_get_laplacian_coeff_vals(dctBlocks):
    laplacianVals = []
    for block in dctBlocks:
        # The AC coefficients are all elements except the first one (DC coefficient)
        ac_vals = block[1:]
        laplacianVals.append(ac_vals)
    return laplacianVals


def decode_get_laplacian_coeff_vals(matrix):
    # Flatten the 8x8 matrix, excluding the DC coefficient (top-left element)
    matrix_flat = [matrix[i][j] for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if not (i == 0 and j == 0)]
    return matrix_flat

def decode_get_gaussian_coeff_vals(matrix):
    # Extract the DC coefficient (top-left element)
    return matrix[0][0]



def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
  
  