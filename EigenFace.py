import Image
import ImageOps
import numpy as np
from numpy import linalg
import matplotlib
from matplotlib import pyplot
import os
import glob
import sys

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r



def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A

    # Note: "lambda" is a Python reserved word


    # compute mean, and subtract mean from every column
    [r,c] = A.shape
    m = np.mean(A,1)
    A = A - np.transpose(np.tile(m, (c,1)))
    B = np.dot(np.transpose(A), A)
    [d,v] = linalg.eig(B)
    # v is in descending sorted order

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
    
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m


def read_train_faces():
    # function faces = read_train_faces()
    # Browse the train directory, read image files and store faces in a matrix
    # train images are supposed to be in "train" subfolder of the current directory
    # faces: face matrix in which each colummn is a colummn vector for 1 face image

    A = []  # A will store list of image vectors

    cwd = os.getcwd()
    train_dir = cwd + "/train"
    
    # browsing the directory
    for infile in glob.glob(os.path.join(train_dir, '*.*')):
        im = Image.open(infile)
        im_arr = np.asarray(im)
        im_arr = im_arr.astype(np.float32)

        # turn an array into vector
        im_vec = np.reshape(im_arr, -1)
        A.append(im_vec)

    faces = np.array(A)
    faces = np.transpose(faces)

    return faces



def float2int8(A):
    # function im = float2int8(A)
    # convert an float array image into grayscale image with contrast from 0-255
    amin = np.amin(A)
    amax = np.amax(A)
    [r,c] = A.shape
    im = ((A - amin) / (amax - amin)) * 255

    return im





