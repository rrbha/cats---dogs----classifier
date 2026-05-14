# ADVANCED LIBRARIES FOR MACHINE LEARNING
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # KNN implementation from the neighbors package
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Evaluation metrics
from sklearn.model_selection import train_test_split # Utility for splitting data into training/testing sets
from sklearn.tree import DecisionTreeClassifier # Implementation of a single Decision Tree
from sklearn.ensemble import RandomForestClassifier # Ensemble learning (collection of decision trees)
from sklearn import svm # Support Vector Machine for non-linear classification
import os
import cv2

# FEATURE EXTRACTION FUNCTION (HISTOGRAM)
# Manual function to extract the 1D Feature Vector (Intensity Histogram)
def getHist(img):
  h,w=img.shape # Get dimensions of the grayscale image
  hist=np.zeros(256) # Initialize a vector of 256 zeros (one for each intensity level)
  for i in range(h):
    for j in range(w):
      c=img[i][j] # Access the intensity of the pixel at (i, j)
      hist[c]=hist[c]+1 # Increment the frequency of that intensity
  return hist # Return the final statistical representation of the image

# DATA PREPARATION & STANDARDIZATION
lbl=[] # List to store the target labels (Cats/Dogs)
dataset=[] # List to store the feature vectors (histograms)

path="dataset/Downloads2"
classes=os.listdir(path)
for i in classes:
  classPath = os.path.join(path,i)
  imgNameList=os.listdir(classPath)
  for j in imgNameList:
   imgPath=os.path.join(classPath,j)
   img = cv2.imread(imgPath)
   # Preprocessing: Convert to Grayscale to simplify features and reduce dimensionality
   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # Resize: Standardize all images to 100x100 to ensure consistent feature vector length
   img = cv2.resize(img, (100, 100))
   # Feature Extraction: Convert the visual matrix into a 1D numerical histogram
   hist=getHist(img) #histogram


   dataset.append(hist) # Add the feature vector to the dataset
   lbl.append(i) # Add the corresponding class label

# Convert lists to NumPy arrays for compatibility with Scikit-Learn models
x = np.array(dataset)
y = np.array(lbl)


#  DATA PARTITIONING
# Splitting the data: 80% for training the models and 20% for testing (validation)
xTrain, xTest, yTrain, yTest=train_test_split(x,y, test_size=0.2, random_state=42)

# MODEL INITIALIZATION & HYPERPARAMETERS
# Initializing multiple classifiers with specific settings
knnModel=KNeighborsClassifier(n_neighbors=3) # KNN with k=3
dtModel=DecisionTreeClassifier(max_depth=3) # DT limited to depth 3 to prevent over-fitting
rfModel=RandomForestClassifier(n_estimators=100) #it's a must to specify number of trees, since rf is a collection of dt
svmModel=svm.SVC(kernel="rbf") #linear / rbf(since images are not linear)

# MODEL TRAINING
# Feeding the training data (Features and Labels) into the models
knnModel.fit(xTrain,yTrain)
dtModel.fit(xTrain,yTrain)
rfModel.fit(xTrain,yTrain)
svmModel.fit(xTrain,yTrain)

# PREDICTION
# Using the trained models to predict labels for the unseen (test) data
yPredictKNN=knnModel.predict(xTest)
yPredictDT=dtModel.predict(xTest)
yPredictRF=rfModel.predict(xTest)
yPredictSVM=svmModel.predict(xTest)


# PERFORMANCE EVALUATION
# Generating Confusion Matrices to visualize True Positives vs. False Positives
conMKNN=confusion_matrix(yTest,yPredictKNN)
conMDT=confusion_matrix(yTest,yPredictDT)
conMRF=confusion_matrix(yTest,yPredictRF)
conMSVM=confusion_matrix(yTest,yPredictSVM)


print("confusion matrix KNN")
print(conMKNN)
print("confusion matrix DT")
print(conMDT)
print("confusion matrix RF")
print(conMRF)
print("confusion matrix SVM")
print(conMSVM)

# Printing detailed classification reports (Precision, Recall, and F1-Score)
reportKNN=classification_report(yTest,yPredictKNN)
reportDT=classification_report(yTest,yPredictDT)
reportRF=classification_report(yTest,yPredictRF)
reportSVM=classification_report(yTest,yPredictSVM)

# Calculating overall Accuracy Scores for comparison
accKNN=accuracy_score(yTest,yPredictKNN)
accDT=accuracy_score(yTest,yPredictDT)
accRF=accuracy_score(yTest,yPredictRF)
accSVM=accuracy_score(yTest,yPredictSVM)

print("(><) (><) (><) (><) (><) (><) (><) (><) (><) (><) (><)")

# DISPLAYING RESULTS
print("accuracy KNN", accKNN)
print("accuracy DT", accDT)
print("accuracy RF", accRF)
print("accuracy SVM", accSVM)

print("(><) (><) (><) (><) (><) (><) (><) (><) (><) (><) (><)")

print("report KNN")
print(reportKNN)
print("report DT")
print(reportDT)
print("report RF")
print(reportRF)
print("report SVM")
print(reportSVM)
