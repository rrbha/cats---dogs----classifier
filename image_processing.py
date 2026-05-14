from google.colab.patches import cv2_imshow # specialized function for displaying images
import matplotlib.pyplot as plt # used for plotting and data visualization
import cv2 # OpenCV: The primary library for image matrix manipulation
import os # os module for directory navigation and file path handling
import numpy as np # NumPy for efficient numerical operations on image arrays

# Setting up the path and identifying the target classes (categories)
path="dataset/Downloads2"
classes=os.listdir(path) # retrieve sub-folders representing different classes
print("Files found in path:", os.listdir(path))
# Iterate through each class and process individual images
for i in classes:
  classPath = os.path.join(path,i) # construct path to the specific class folder
  imgNameList=os.listdir(classPath) # list all filenames within the class directory
  for j in imgNameList:
   imgPath=os.path.join(classPath,j) # Create full path for the specific image file
   img = cv2.imread(imgPath) # load image into a 3D Numerical Matrix (rows, cols, channels)

   img = cv2.resize(img, (200, 200))

   # Accessing image dimensions
   # as studied, row=Height, col=Width, lay=Color Channels (B,G,R)
   row=img.shape[0]
   col=img.shape[1]
   lay=img.shape[2]

   cv2_imshow(img) # Display original image before any enhancement

   # Initialize empty matrices (Zeros) to store the results of filtered images
   newImg=np.zeros((row,col,lay))
   newImgkernal=np.zeros((row,col,lay))


   # MANUAL BRIGHTNESS ADJUSTMENT
   # Implementing pixel-by-pixel intensity modification using triple-nested loops
   for r in range(row):
     for c in range(col):
        for k in range(lay):
          # Increment pixel intensity by a constant
          value = int(img[r,c,k]) + 20
          # Ensure intensities remain within the valid [0, 255] range
          if value > 255:
                img[r,c,k] = 255
          else:
                img[r,c,k] = value

   cv2_imshow(img) # display result after point-wise brightness enhancement

   # IMAGE SHARPENING (Method 1: Manual Neighbor Calculation)
   # directly calculating the High-Pass output using spatial offsets (Center vs Neighbors)
   for r in range(1,row-1):
     for c in range(1,col-1):
       for k in range(lay):
        # Define neighbors: top, bottom, left, and right
         center=img[r,c,k]
         top=img[r-1,c,k]
         bottom=img[r+1,c,k]
         left=img[r,c-1,k]
         right=img[r,c+1,k]
         # Manual Sharpness Formula: Enhancing local contrast
         v=5*center-top-bottom-left-right
         # Apply manual clipping to prevent overflow or negative values
         if v>255:
           v=255
         elif v<0:
          v=0

         newImg[r,c,k]=v


# (Method 2: Kernel-Based Convolution)
# Define a 3x3 High-Pass Kernel for Sharpening
   sharpen_kernel=np.array(
       [[0,-1,0],
       [-1,5,-1],
       [0,-1,0]]
                   )

  # Define a 3x3 Averaging Kernel for Blurring/Smoothing
   blur_kernel=np.array(
       [
           [1/9,1/9,1/9],
           [1/9,1/9,1/9],
           [1/9,1/9,1/9]
       ]
   )


# Applying Sharpening Kernel
# Implementing spatial convolution through sliding window matrix multiplication
   for r in range(1,row-1):
     for c in range(1,col-1):
      for k in range(lay):
        # Extract the 3x3 local neighborhood (Region of Interest)
        region=img[r-1:r+2,c-1:c+2,k]
        # Multiply region by the kernel and sum to get the new pixel value
        newValue=region*sharpen_kernel
        newImgkernal[r,c,k]=np.sum(newValue)

# Final normalization using clipping and display
   newImgkernal=np.clip(newImgkernal,0,255)
   cv2_imshow(newImgkernal)

# Applying Blurring Kernel (Smoothing)
# Using the same convolution logic with the Averaging Filter
   for r in range(1,row-1):
     for c in range(1,col-1):
      for k in range(lay):
        region=img[r-1:r+2,c-1:c+2,k]
        newValue=region*blur_kernel
        newImgkernal[r,c,k]=np.sum(newValue)

# Final clipping and display of the blurred image result
   newImgkernal=np.clip(newImgkernal,0,255)
   cv2_imshow(newImgkernal)
