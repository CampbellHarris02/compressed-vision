import numpy as np
import os
import json



# Creates a NxN DCT transform matrix, which applies a 1D DCT transform
# to a vector of 1D signals. See dct_2d_matrix for 2D DCT.
def dct_matrix(N):
    """Create an NxN DCT transformation matrix."""
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = np.sqrt(1 / N)
            else:
                C[i, j] = np.sqrt(2 / N) * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return C

# Combines two 1D DCT matricies using the Kroneker product
def dct_2d_matrix(dctMatrix):
    return np.kron(dctMatrix, dctMatrix)

# Actually compute the 2D DCT transform on a block. Note that the kroneker
# product gives a N^2 by N^2 matrix as the function, and expects the input to
# be a flattened block (which will become a 1 by N^2 vector instead of a N by N
# matrix)
def dct_2d(flattenedSignal, dctKronMatrix):
    return dctKronMatrix @ flattenedSignal

def apply_dct(imageBlock, lookUpTableFileName, rewrite=False):
    LUT = []
    dctBlocks = []
    dctTransform = []
    shape = 8
    if (os.path.exists(lookUpTableFileName) and not rewrite):
        with open(lookUpTableFileName, 'r') as file:
            LUT = np.array(json.load(file))
    else:
        LUT = dct_matrix(shape)
        LUT = dct_2d_matrix(LUT)
        with open(lookUpTableFileName, 'w') as file:
            json.dump(LUT.tolist(), file)

    dctTransform = dct_2d(imageBlock.flatten(), LUT)
    dctTransform = dctTransform.reshape(shape, shape)
    return dctTransform


def apply_idct(dctBlock, lookUpTableFileName, rewrite=False):
    LUT = []
    imageBlocks = []
    idctTransform = []
    shape = 8
    if (os.path.exists(lookUpTableFileName) and not rewrite):
        with open(lookUpTableFileName, 'r') as file:
            LUT = np.array(json.load(file))
            LUT = np.linalg.inv(LUT)
    else:
        LUT = dct_matrix(shape)
        LUT = dct_2d_matrix(LUT)
        with open(lookUpTableFileName, 'w') as file:
            json.dump(LUT.tolist(), file)
        LUT = np.linalg.inv(LUT)
    
    idctTransform = dct_2d(dctBlock.flatten(), LUT)
    idctTransform = idctTransform.reshape(shape, shape)
    
    return idctTransform








import numpy as np
import os
import json

# Creates a NxN DCT transform matrix, which applies a 1D DCT transform
# to a vector of 1D signals. See dct_2d_matrix for 2D DCT.
def dct_1d_matrix(N):
    """Create an NxN DCT transformation matrix."""
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = np.sqrt(1 / N)
            else:
                C[i, j] = np.sqrt(2 / N) * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))
    return C

# Combines two 1D DCT matricies using the Kroneker product
def dct_2d_matrix(dctMatrix):
    return np.kron(dctMatrix, dctMatrix)

# If file does NOT exist, it will be created. If it does exist, it will used unless asked to overwrite it.
def get_lookup_table(fileName, dimension=8, overwriteFile=False):
    LUT = []
    if (not os.path.exists(fileName) or overwriteFile):
        LUT = dct_1d_matrix(dimension)
        LUT = dct_2d_matrix(LUT)
        with open(fileName, 'w') as file:
            json.dump(LUT.tolist(), file)
    else:
        with open(fileName, 'r') as file:
            LUT = np.array(json.load(file))
    return LUT

def dct_2d(flattenedSignal, dctKronMatrix):
    return dctKronMatrix @ flattenedSignal

def apply_dct(imageBlocks, LUT):
    # dctBlocks = []
    # dctTransform = []
    # shape = imageBlocks[0].shape
    # if (shape[0] != shape[1]):
    #     print("ERROR: IMAGE BLOCKS NOT SQUARE")
    # else:
    #     shape = shape[0]
    # for block in imageBlocks:
    #     dctTransform = dct_2d(block.flatten(), LUT)
    #     dctTransform = dctTransform.reshape(shape, shape)
    #     dctBlocks.append(dctTransform)
    # return dctBlocks

    blockSize = 8
    dctBlocks = np.zeros_like(imageBlocks, dtype=np.float32)
    # Loop over all blocks and apply DCT
    dctBlocks = dct_2d(imageBlocks.flatten(), LUT).reshape(blockSize, blockSize)
    return dctBlocks

def write_dct_to_json_file(dctBlocks, fileName):
    dctBlocksList = dctBlocks.tolist()
    with open(fileName, 'w') as jsonFile:
        json.dump(dctBlocksList, jsonFile)

def read_dct_from_json_file(fileName):
    """ Load the NumPy array from a JSON file. """
    with open(fileName, 'r') as jsonFile:
        dctBlocksList = json.load(jsonFile)
    # Convert the list of lists back into a NumPy array
    return np.array(dctBlocksList)