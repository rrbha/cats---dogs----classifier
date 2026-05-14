from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

path="dataset/Downloads2"
classes=os.listdir(path)
for i in classes:
  classPath = os.path.join(path,i)
  imgNameList=os.listdir(classPath)
  for j in imgNameList:
   imgPath=os.path.join(classPath,j)
   img = cv2.imread(imgPath) # Loading the image as a BGR Numerical Matrix
   # Extract image dimensions (Rows, Columns, and Color Layers)
   row=img.shape[0]
   col=img.shape[1]
   lay=img.shape[2]

   cv2_imshow(img) # Display original image before analysis

   # HISTOGRAM INITIALIZATION
   # Initialize three frequency arrays (one for each color channel: R, G, B)
   # Each array has 256 bins representing intensity levels from 0 to 255
   hr=np.zeros(256)
   hg=np.zeros(256)
   hb=np.zeros(256)

# we visit every pixel to calculate the intensity distribution
   for r in range(row):
    for c in range(col):
      # Accessing color channels (OpenCV uses BGR order)
      ri=img[r,c,2] # Red Intensity (Index 2)
      gi=img[r,c,1] # Green Intensity (Index 1)
      bi=img[r,c,0] # Blue Intensity (Index 0)

     # Increment the statistical counter for each specific intensity value
     # This turns the visual content into a numerical frequency vector
      hr[ri]+=1
      hg[gi]+=1
      hb[bi]+=1

   #  DATA VISUALIZATION
   # Plotting the histograms to analyze the color distribution (Numerical Representation)
    plt.plot(hr) # Visualizing the Red channel distribution
    plt.plot(hg) # Visualizing the Green channel distribution
    plt.plot(hb) # Visualizing the Blue channel distribution
    plt.show()
