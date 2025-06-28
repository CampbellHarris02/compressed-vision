



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import json
import random
from random import randrange

from dct import apply_dct, get_lookup_table
import image_formatting as imf










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


def create_montage(blocks, h, w, block_size):
    """ Create a montage of blocks and ensure the final size is the same as the original image """
    montage_image = np.zeros((h, w))
    block_idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = blocks[block_idx]
            # Ensure block fits inside the image boundaries
            block_height = min(block_size, h - i)
            block_width = min(block_size, w - j)
            montage_image[i:i+block_height, j:j+block_width] = block[:block_height, :block_width]
            block_idx += 1
    return montage_image


def normalize_image(image):
    """ Normalize the image for better visualization """
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)  # Scale to 0-1 range


class DCT:
    """ This class can be used for both the DCT and inverse DCT """

    def __init__(self, signal, N):
        self.signal = signal  # 2-dimensional array
        self.N = N

    def alpha(self, u):
        """ DCT scaling factor """
        if u == 0:
            return np.sqrt(1 / self.N)
        else:
            return np.sqrt(2 / self.N)

    def c(self, u, v):
        """ Perform DCT on the signal using the given formula """
        constant = self.alpha(u) * self.alpha(v)
        temp_sum = 0
        for x in range(self.N):  # 0, 1,..., N-1
            for y in range(self.N):
                temp_sum += self.signal[x, y] * np.cos(((2*x + 1) * u * np.pi) / (2 * self.N)) * np.cos(((2*y + 1) * v * np.pi) / (2 * self.N))
        result = constant * temp_sum
        return result

    def f(self, x, y):
        """ Perform inverse DCT on the signal using the given formula """
        temp_sum = 0
        for u in range(self.N):  # 0, 1,..., N-1
            for v in range(self.N):
                temp_sum += self.alpha(u) * self.alpha(v) * self.signal[u, v] * np.cos(((2*x + 1) * u * np.pi) / (2 * self.N)) * np.cos(((2*y + 1) * v * np.pi) / (2 * self.N))
        return temp_sum

    def dct_block(self):
        """ Return the DCT of the input block """
        rows, cols = self.signal.shape
        signal_transform = np.zeros((rows, cols))
        for u in range(rows):
            for v in range(cols):
                signal_transform[u, v] = self.c(u, v)
        return signal_transform

    def inv_dct_block(self):
        """ Return the inverse DCT of the input block """
        rows, cols = self.signal.shape
        signal_transform = np.zeros((rows, cols))
        for x in range(rows):
            for y in range(cols):
                signal_transform[x, y] = self.f(x, y)
        return signal_transform



def zigzag_order(block):
    """
    Traverse an 8x8 block in zigzag order and return a flattened array.
    """
    rows, cols = block.shape
    result = []
    for diagonal in range(rows + cols - 1):
        if diagonal < cols:
            row = 0
            col = diagonal
        else:
            row = diagonal - cols + 1
            col = cols - 1

        while row < rows and col >= 0:
            result.append(block[row, col])
            row += 1
            col -= 1
    return np.array(result)

import numpy as np

def inverse_zigzag_order(flat_array, block_size=8):
    # Initialize an empty 8x8 block
    block = np.zeros((block_size, block_size), dtype=flat_array.dtype)
    rows, cols = block.shape
    # Index to track the position in the flat_array
    flat_index = 0
    # Traverse the block in zigzag order and fill it with values from flat_array
    for diagonal in range(rows + cols - 1):
        if diagonal < cols:
            row = 0
            col = diagonal
        else:
            row = diagonal - cols + 1
            col = cols - 1

        while row < rows and col >= 0:
            block[row, col] = flat_array[flat_index]
            flat_index += 1
            row += 1
            col -= 1
    return block


def quantize_scalar(value, codebook):
    """
    Given a single float 'value' and a 1D array 'codebook',
    return the nearest codebook entry.
    """
    # Find the index of the nearest codebook value, NEAREST NEIGHBOUR CONDITION
    idx = np.argmin((codebook - value)**2)
    return codebook[idx], idx



def quantize_block_dct(block_values, codebooks_dict, noise_idx, B_bits=24):
    block_size = 8

    if B_bits == 24:  # zigzag of B 24 bit allocation table
        codebook_selector = [
            8, 8, 8, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 58:  # zigzag of B 58 bit allocation table
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 76:
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 3, 4, 4, 4, 3, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    else:  # there should only exist the options above
        print("ERROR")
        return None, None

    block_flat = []
    block_indexes_flat = []

    for i in range(len(codebook_selector)):
        if codebook_selector[i] == 0:
            break
        # Quantize the value
        if i == 0:  # DC Coefficient
            dc_encoded, dc_idx = quantize_scalar(block_values[i], codebooks_dict[f"dc_rate_{codebook_selector[i]}_set_{noise_idx}"])
            block_flat.append(dc_encoded)
            # Convert DC index to binary array of length codebook_selector[i]
            binary_dc_idx = [int(bit) for bit in f"{dc_idx:0{codebook_selector[i]}b}"]
            block_indexes_flat.extend(binary_dc_idx)  # Use extend instead of append
        else:  # AC Coefficients
            ac_encoded, ac_idx = quantize_scalar(block_values[i], codebooks_dict[f"ac_rate_{codebook_selector[i]}_set_{noise_idx}"])
            block_flat.append(ac_encoded)
            # Convert AC index to binary array of length codebook_selector[i]
            binary_ac_idx = [int(bit) for bit in f"{ac_idx:0{codebook_selector[i]}b}"]
            block_indexes_flat.extend(binary_ac_idx)  # Use extend instead of append

    # Return the flat block and the binary indexes of the flat block
    return block_flat, block_indexes_flat



def create_normalized_blocks(normalizedDcVals, normalizedAcVals, block_size=8):
    num_blocks = len(normalizedDcVals)
    norm_dct_blocks = []

    for i in range(num_blocks):
        # Get the normalized DC and AC values for the current block
        dc_val = normalizedDcVals[i]
        ac_vals = normalizedAcVals[i]  # Each row in normalizedAcVals corresponds to a block

        # Ensure dc_val is a 1D array with a single element
        dc_val = np.array([dc_val]).flatten()  # Convert to 1D array

        # Ensure ac_vals is a 1D array
        ac_vals = np.array(ac_vals).flatten()

        # Combine DC and AC values into a single 1D array of shape (64,)
        block_values = np.concatenate((dc_val, ac_vals))
        norm_dct_blocks.append(block_values)

    # Convert to a NumPy array of shape (num_blocks, 64)
    return np.array(norm_dct_blocks)



def compress_image_with_codebook(image_array, codebooks_dict, B_bits, noise_idx, block_size):
    """
    1) Break the image into blocks of size 'block_size x block_size'.
    2) For each block, do DCT -> quantize with codebook
    3) return the string of indexes
    """
    # Break image into blocks
    blocks = blockify_image(image_array, block_size)
    h, w = image_array.shape
    
    DCT_LOOKUP_TABLE_FILE_NAME = 'dctLookupTable.json'
    dctLookupTable = get_lookup_table(DCT_LOOKUP_TABLE_FILE_NAME)

    # Compress each block with DCT + codebook quantization
    compressed_block_indexes = []
    dct_blocks = []
    
    for block in blocks:
            # 2D DCT
        block_dct = apply_dct(block, dctLookupTable)
        block_values = zigzag_order(block_dct)
        dct_blocks.append(block_values)
        
    gaussianCoeffVals = np.array(imf.encode_get_gaussian_coeff_vals(dct_blocks))
    dcMean            = np.mean(gaussianCoeffVals)
    dcStd             = np.std(gaussianCoeffVals)
    normalizedDcVals  = (gaussianCoeffVals - dcMean) / dcStd
    
    laplaciancoeffvals  = np.array(imf.encode_get_laplacian_coeff_vals(dct_blocks))
    acStd               = np.std(laplaciancoeffvals)
    targetStd           = 1
    normalizedAcVals    = laplaciancoeffvals * (targetStd / acStd)
    
    norm_blocks = create_normalized_blocks(normalizedDcVals, normalizedAcVals)
    
    norm_dct_blocks = []
    for block in norm_blocks:
        # Convert block to float if not already
        block_float = block.astype(np.float32)
        block_flat, block_indexes_flat = quantize_block_dct(block_float, codebooks_dict, noise_idx, B_bits)
        # apply block compression
        compressed_block_indexes.extend(block_indexes_flat)
        norm_dct_blocks.append(block_flat)
        
    return compressed_block_indexes, dcMean, dcStd, acStd




def dequantize_image(compressed_img_array, codebooks_dict, B_bits, noise_idx, block_size, dcMean, dcStd, acStd):
    # Split the compressed image array into binary blocks of size B_bits
    binary_blocks = [compressed_img_array[i:i + B_bits] for i in range(0, len(compressed_img_array), B_bits)]

    dctLookupTable      = get_lookup_table('dctLookupTable.json')
    idctLookupTable     = np.linalg.inv(dctLookupTable)

    if B_bits == 24:  # zigzag of B 24 bit allocation table
        codebook_selector = [
            8, 8, 8, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 58:  # zigzag of B 58 bit allocation table
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 76:
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 3, 4, 4, 4, 3, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    else:  # there should only exist the options above
        print("ERROR")
        return None, None

    # Process each binary block
    decoded_blocks = []
    for bin_block in binary_blocks:
        processed_block = []
        start_idx = 0
        for selector in codebook_selector:
            if selector == 0:
                processed_block.append(0)
            else:
                sublist = bin_block[start_idx:start_idx + selector]
                binary_str = ''.join(map(str, sublist))
                index_integer_value = int(binary_str, 2)
                if start_idx == 0:  # dc coefficient
                    integer_value = codebooks_dict[f"dc_rate_{selector}_set_{noise_idx}"][index_integer_value]
                else:  # ac coefficient
                    integer_value = codebooks_dict[f"ac_rate_{selector}_set_{noise_idx}"][index_integer_value]
                processed_block.append(integer_value)
                start_idx += selector

        # Inverse zigzag to get the 8x8 block
        norm_block = inverse_zigzag_order(np.array(processed_block), block_size)

        # Extract DC and AC values
        dc_val = (imf.decode_get_gaussian_coeff_vals(norm_block) * dcStd) + dcMean
        ac_vals = imf.decode_get_laplacian_coeff_vals(norm_block)
        ac_vals = np.array(ac_vals) / (1 / acStd)  # Reverse normalization for AC values

        # Combine DC and AC values into a single block
        block = np.zeros((block_size, block_size), dtype=float)
        block[0][0] = dc_val
        index = 0
        for i in range(block_size):
            for j in range(block_size):
                if i == 0 and j == 0:
                    continue  # Skip DC coefficient
                block[i][j] = ac_vals[index]
                index += 1

        # Inverse DCT transform
        img_block = apply_dct(block, idctLookupTable)
        img_block = np.clip(img_block, 0, 255)

        decoded_blocks.append(img_block)

    return decoded_blocks


def compute_psnr(original_array, reconstructed_array):
    mse = np.mean((original_array - reconstructed_array) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr



#result of modulo 2 adding x1 and x2
def modulo_add(x1, x2):
  if x1 == x2:
    return 0
  else:
    return 1


  
def conditional_prob(b, j, i, delta, epsilon, epsilon_prime, Q):
  result = 0
  if (b[j, 0] == 2):
    #result = result + epsilon_prime
    result = result + epsilon_prime
  else:
    #result = result + (epsilon**(modulo_add(b[j, 0], b[i, 0])))*((1 - epsilon - epsilon_prime)**(1 - modulo_add(b[j, 0], b[i, 0])))
    result = result + (epsilon**(modulo_add(b[j, 0], b[i, 0])))*((1 - epsilon - epsilon_prime)**(1 - modulo_add(b[j, 0], b[i, 0])))
  for i in range(1, b.shape[1]):
    if(b[j, k] == 2): #COLUMN 2 OF Q MATRIX
      if(b[j, k-1] == 2):
        #result = result*(epsilon_prime + delta)/(1 + delta)
        result = result*Q[2, 2] #ENTRY IN THE DIAGONAL
      elif(b[j, k-1] == 0):
        #result = result*(epsilon_prime)/(1 + delta)
        result = result*Q[0, 2] #ENTRY IN THAT COLUMN BUT NOT THE DIAGONAL
      elif(b[j, k-1] == 1):
        #result = result*(epsilon_prime)/(1 + delta)
        result = result*Q[0, 1] #ENTRY IN THAT COLUMN BUT NOT THE DIAGONAL
    elif(modulo_add(b[j, k], b[i, k]) == 1): #COLUMN 1 OF Q MATRIX
      if(modulo_add(b[j, k-1], b[i, k-1]) == 1):
        #result = result*(epsilon + delta)/(1 + delta)
        result = result*Q[1, 1] #ENTRY IN THE DIAGONAL
      elif(modulo_add(b[j, k-1], b[i, k-1]) == 0):
        #result = result*(epsilon)/(1 + delta)
        result = result*Q[0, 1] #ENTRY IN THAT COLUMN BUT NOT THE DIAGONAL
      elif(modulo_add(b[j, k-1], b[i, k-1]) == 2):
        #result = result*(epsilon)/(1 + delta)
        result = result*Q[2, 1] #ENTRY IN THAT COLUMN BUT NOT THE DIAGONAL
    elif(modulo_add(b[j, k], b[i, k]) == 0): #COLUMN 0 OF Q MATRIX
      if(modulo_add(b[j, k-1], b[i, k-1]) == 0):
        #result = result*(1 - epsilon_prime - epsilon + delta)/(1 + delta)
        result = result*Q[0, 0] #ENTRY IN THE DIAGONAL
      elif(modulo_add(b[j, k-1], b[i, k-1]) == 1):
        #result = result*(1 - epsilon_prime - epsilon)/(1 + delta)
        result = result*Q[1, 0] #ENTRY IN THAT COLUMN BUT NOT THE DIAGONAL
      elif(modulo_add(b[j, k-1], b[i, k-1]) == 2):
        #result = result*(1 - epsilon_prime - epsilon)/(1 + delta)
        result = result*Q[2, 0] #ENTRY IN THAT COLUMN BUT NOT THE DIAGONAL
  return result




# "Training" or optimization-related parameters (NOT used by the Polya channel):
T_0 = 10
alpha = 0.97
T_f = 0.00025
quantization_rate = 3
N_fail = 50000
N_success = 5
N_cut = 200
k = 1
distribution = 'laplace'
num_centroids = 8





def simulate_channel_string(len_string, R, G, B, M, Del, debug=False):
    """
    Simulate a bursty bit-flip channel via a Polya urn process.

    Internal "colors":
       - 'B' -> we record a 0
       - 'R' -> we record a 1
       - 'G' -> we also record a 1 (no 2 in the output)

    Args:
    -------
    len_string : int
        Number of picks (output bits) to generate.
    R, G, B : int
        Initial counts of Red, Green, and Blue in the urn.
    M : int
        The size of the memory (number of past picks to store).
    Del : int
        Number of balls to add or replace each step.
    debug : bool
        If True, prints debug info for each iteration.

    Returns:
    -------
    ret_string : list of int
        The sequence of picks (each 0 or 1).
    """
    # Track counts of each color in the urn
    count_R = R
    count_G = G
    count_B = B
    
    # Memory of the last M picks (store color labels here, e.g. 'R', 'G', 'B')
    memory = [None] * M
    
    # The output bit sequence
    ret_string = []
    
    for i in range(len_string):
        # Total number of balls in the urn
        total = count_R + count_G + count_B
        
        # Pick a random index in the combined "urn"
        pick_idx = randrange(total)
        
        # Decide which color was picked based on pick_idx
        if pick_idx < count_B:
            pick_ball = 'B'
            ret_string.append(0)     # B -> 0
        elif pick_idx < count_B + count_R:
            pick_ball = 'R'
            ret_string.append(1)     # R -> 1
        else:
            pick_ball = 'G'
            ret_string.append(1)     # G -> 1 (no 2 in output)
        
        # Remove the oldest from memory, add the newest
        discard_ball = memory.pop(0)
        memory.append(pick_ball)
        
        # Update the urn based on Polya logic
        if discard_ball is None:
            # If discarding None, just add Del copies of the picked color
            if pick_ball == 'B':
                count_B += Del
            elif pick_ball == 'R':
                count_R += Del
            else:  # pick_ball == 'G'
                count_G += Del
        elif discard_ball != pick_ball:
            # We replace up to Del copies of discard_ball with pick_ball
            if discard_ball == 'B':
                replaced = min(count_B, Del)
                count_B -= replaced
                if pick_ball == 'R':
                    count_R += replaced
                else:  # pick_ball == 'G'
                    count_G += replaced
                
            elif discard_ball == 'R':
                replaced = min(count_R, Del)
                count_R -= replaced
                if pick_ball == 'B':
                    count_B += replaced
                else:  # pick_ball == 'G'
                    count_G += replaced
                
            else:  # discard_ball == 'G'
                replaced = min(count_G, Del)
                count_G -= replaced
                if pick_ball == 'B':
                    count_B += replaced
                else:  # pick_ball == 'R'
                    count_R += replaced
        
        if debug:
            print(f"Iteration {i}")
            print(f"  Picked color:  {pick_ball}")
            print(f"  Discard color: {discard_ball}")
            print(f"  Urn counts -> B:{count_B}, R:{count_R}, G:{count_G}")
            print(f"  Memory:     {memory}")
            print(f"  ret_string: {ret_string}")
            print("---------------")
    
    return ret_string






def implement_channel(image_string, R, G, B, M, Del):
#run the channel on a given string
  channel_string = simulate_channel_string(len(image_string), R, G, B, M, Del)
  ret_string = [None] * len(channel_string)
  for i in range(len(channel_string)):
    if (channel_string[i] == 2):
      ret_string[i] = 2
    else:
      ret_string[i] = modulo_add(image_string[i], channel_string[i])
  return ret_string












#### LLOYD MAX QUANTIZER

def quantize_block_dct_lloydmax(block_values, codebooks_dict, B_bits=24):
    """
    Given a block of DCT coefficients (block_values, length up to 64),
    quantize DC and AC parts using a Lloyd–Max codebook.

    codebooks_dict is a dictionary containing:
      - "dc_lloydmax_rate_{bits}" : array of DC centroids
      - "ac_lloydmax_rate_{bits}" : array of AC centroids

    B_bits selects the bit-allocation pattern for up to 64 coefficients
    (or fewer if your table has 0 for no bits).
    """
    block_size = 8

    # 1) Choose bit allocation pattern
    if B_bits == 24:
        codebook_selector = [
            8, 8, 8, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 58:
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 76:
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 3, 4, 4, 4, 3, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    else:
        print("ERROR: Unsupported B_bits")
        return None, None

    block_flat = []
    block_indexes_flat = []

    # 2) For each coefficient in block_values, quantize using the specified bits
    for i, bits in enumerate(codebook_selector):
        if bits == 0:
            break  # no more bits allocated
        # If i == 0, we treat it as DC; else AC
        if i == 0:
            # DC Coeff
            dc_codebook = codebooks_dict[f"dc_rate_{bits}_lloydmax_codebook"]
            dc_encoded, dc_idx = quantize_scalar(block_values[i], dc_codebook)

            block_flat.append(dc_encoded)

            # Convert DC index to binary array
            bin_str = f"{dc_idx:0{bits}b}"
            block_indexes_flat.extend([int(bit) for bit in bin_str])
        else:
            # AC Coeff
            ac_codebook = codebooks_dict[f"ac_rate_{bits}_lloydmax_codebook"]
            ac_encoded, ac_idx = quantize_scalar(block_values[i], ac_codebook)

            block_flat.append(ac_encoded)

            # Convert AC index to binary array
            bin_str = f"{ac_idx:0{bits}b}"
            block_indexes_flat.extend([int(bit) for bit in bin_str])

    return block_flat, block_indexes_flat



def compress_image_with_lloydmax_codebook(image_array, codebooks_dict, B_bits, block_size=8):
    """
    1) Break the image into 8x8 blocks.
    2) DCT + Zigzag.
    3) Normalization (DC & AC).
    4) Lloyd–Max quantize each block using 'quantize_block_dct_lloydmax'.
    5) Return bitstream + stats.
    """
    # Break image into blocks
    blocks = blockify_image(image_array, block_size)
    h, w = image_array.shape
    
    DCT_LOOKUP_TABLE_FILE_NAME = 'dctLookupTable.json'
    dctLookupTable = get_lookup_table(DCT_LOOKUP_TABLE_FILE_NAME)

    # Compress each block with DCT + codebook quantization
    compressed_block_indexes = []
    dct_blocks = []
    
    for block in blocks:
            # 2D DCT
        block_dct = apply_dct(block, dctLookupTable)
        block_values = zigzag_order(block_dct)
        dct_blocks.append(block_values)
        
    gaussianCoeffVals = np.array(imf.encode_get_gaussian_coeff_vals(dct_blocks))
    dcMean            = np.mean(gaussianCoeffVals)
    dcStd             = np.std(gaussianCoeffVals)
    normalizedDcVals  = (gaussianCoeffVals - dcMean) / dcStd
    
    laplaciancoeffvals  = np.array(imf.encode_get_laplacian_coeff_vals(dct_blocks))
    acStd               = np.std(laplaciancoeffvals)
    targetStd           = 1
    normalizedAcVals    = laplaciancoeffvals * (targetStd / acStd)
    
    norm_blocks = create_normalized_blocks(normalizedDcVals, normalizedAcVals)
    
    # For each 8x8 block (after normalization):
    compressed_block_indexes = []
    norm_dct_blocks = []
    for block in norm_blocks:
        block_float = block.astype(np.float32)

        # ---- Lloyd–Max version
        block_flat, block_indexes_flat = quantize_block_dct_lloydmax(
            block_float,
            codebooks_dict,
            B_bits
        )
        
        compressed_block_indexes.extend(block_indexes_flat)
        norm_dct_blocks.append(block_flat)

    return compressed_block_indexes, dcMean, dcStd, acStd



def dequantize_image_lloydmax(
    compressed_img_array, codebooks_dict, 
    B_bits, block_size, dcMean, dcStd, acStd
):
    """
    Dequantize an image using a Lloyd–Max codebook for DC/AC.

    Args:
        compressed_img_array: 1D list/array of bits (0 or 1), the compressed bitstream.
        codebooks_dict: Dict containing e.g.:
            {
              "dc_lloydmax_rate_8": [...],  # 2^8 centroids
              "ac_lloydmax_rate_7": [...],  # 2^7 centroids
               ...
            }
        B_bits: total bits allocated for one block in 'codebook_selector'
        block_size: typically 8 (for 8x8 DCT blocks)
        dcMean, dcStd: used for un-normalizing the DC coefficient
        acStd: used for un-normalizing the AC coefficients

    Returns:
        decoded_blocks: list of 2D arrays (shape block_size x block_size) 
                        representing the reconstructed image blocks.
    """
    # Split the compressed bit array into chunks of size B_bits
    binary_blocks = [
        compressed_img_array[i : i + B_bits] 
        for i in range(0, len(compressed_img_array), B_bits)
    ]

    # Load your DCT lookup tables (assuming you have them)
    dctLookupTable  = get_lookup_table('dctLookupTable.json')
    idctLookupTable = np.linalg.inv(dctLookupTable)

    # Choose your bit allocation table
    if B_bits == 24:
        codebook_selector = [
            8, 8, 8, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 58:
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    elif B_bits == 76:
        codebook_selector = [
            8, 7, 7, 6, 6, 6, 4, 5,
            5, 4, 3, 4, 4, 4, 3, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
    else:
        print("ERROR: B_bits not recognized")
        return None

    decoded_blocks = []

    # For each chunk of bits
    for bin_block in binary_blocks:
        processed_block = []
        start_idx = 0
        
        # For each coefficient's bit allocation
        for bits in codebook_selector:
            if bits == 0:
                # No bits => coefficient is 0
                processed_block.append(0)
                continue

            # Extract 'bits' from bin_block
            sublist = bin_block[start_idx : start_idx + bits]

            # Convert to integer index
            binary_str = ''.join(map(str, sublist))
            index_integer_value = int(binary_str, 2)

            # If this is the first coefficient (DC)
            if start_idx == 0:
                # Retrieve from DC Lloyd–Max codebook
                key = f"dc_rate_{bits}_lloydmax_codebook"
                integer_value = codebooks_dict[key][index_integer_value]
            else:
                # Retrieve from AC Lloyd–Max codebook
                key = f"ac_rate_{bits}_lloydmax_codebook"
                integer_value = codebooks_dict[key][index_integer_value]

            processed_block.append(integer_value)
            start_idx += bits

        # 1D -> 8x8 block using inverse zigzag
        norm_block = inverse_zigzag_order(np.array(processed_block), block_size)

        # Re-scale DC
        dc_val = (imf.decode_get_gaussian_coeff_vals(norm_block) * dcStd) + dcMean

        # Re-scale AC
        ac_vals = imf.decode_get_laplacian_coeff_vals(norm_block)
        ac_vals = np.array(ac_vals) / (1 / acStd)  # inverse of multiplying by (targetStd/acStd)

        # Combine DC + AC into a full 8x8 block
        block = np.zeros((block_size, block_size), dtype=float)
        block[0][0] = dc_val
        idx = 0
        for i in range(block_size):
            for j in range(block_size):
                if i == 0 and j == 0:
                    continue
                block[i][j] = ac_vals[idx]
                idx += 1

        # Inverse DCT
        img_block = apply_dct(block, idctLookupTable)
        img_block = np.clip(img_block, 0, 255)

        decoded_blocks.append(img_block)

    return decoded_blocks