# Pattern-Recognition
This repository is opened for 2023-2024 Spring Pattern Recognition techical elective course. All individual assignment and final group project that were given in the course by lecturer are added to the repository. And their content is explained in the readme file.

## ASSIGNMENTS 
 ### HOMEWORK 1
These tasks involve basic image processing techniques like color space transformations, flipping, rotating, and resizing images

  Task 1: 
In this task, you need to perform color transformation on an image using RGB to YIQ and YIQ to RGB conversions with for loops. 
  1. Convert the image from RGB to YIQ using for loops.
  2. Then, convert the image from YIQ to RGB using for loops.

   Task 2: 
For image rotations and resizing:
1. Flip the cat and dog images vertically.
2. Flip the cat and dog images horizontally.
3. Rotate the cat and dog images to the left by 90 degrees.
4. Rotate the cat and dog images to the right by 90 degrees.
5. Resize the cat and dog images to half while keeping the aspect ratio.
6. Finally, display both the input and output images.

 ### HOMEWORK 2 : KNN Classification with Custom Similarity
This homework solution file contains Python code to perform K-nearest neighbors (KNN) classification using custom similarity for the CIFAR-10 dataset.

#### Steps:

1. **Download the CIFAR-10 dataset**:
```
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
3. **Convert images to vector format**:
   - Convert images to vector format where each image with dimensions 32x32x3 is converted to a 1x3072 vector.
4. **Custom Similarity Function**:
   - Write a function that calculates similarity between a test sample and training samples using a custom similarity function. This function should return the class name with the highest similarity among the k nearest neighbors.
![image](https://github.com/bengisu-sahin/Pattern-Recognition/assets/71591780/88c10605-3f00-448b-9c40-f6d5407a6013)
5. **Test Code**:
   - Define a test sample (`sample_test`) from `x_test`.
   - Choose the value of `k` (number of nearest neighbors).
   - Use the `knnCustomSimilarity` function to classify the test sample.
   - I used the test code in below
```
sample_test = x_test[19,:]
k=5
print("Actual Class Label:", y_test[19])
print("Actual Class Label name corresponding to the main class: ",categories[y_test[19][0]])
similar_class_name = knnCustomSimilarity(x_train, y_train, sample_test, k )
print("Predicted Class Label:", similar_class_name)
print("Predicted Class Label name corresponding to the main class: ",categories[similar_class_name[0]])
display_image(sample_test)
```
 ### HOMEWORK 3 : KNN Classification with Custom Similarity
#### Part 1: 
In this HW, you are expected to make an experiment with L1-distance classifier on Principal Component Features (PCA) features, explained in class. Recall the previous lab, in like manner, the experiment will be conducted on the real life problem, called a classification problem among multi categories. The experiment is about Caltech-101 datasets, there are 15 classes, and each one contains different number of samples. 
The aim is to analyze the discriminative capability of PCA features for a classification problem. After extraction PCA features, we will train with L1-distance classifier by oneagainst-all methodology. According to one-against-all methodology, for L1-classifer, it means that the lowestdistance score refers to predicted target class of processed sample. 

#### Part 2: Feature Extraction and Classification Stages
 **1. Stage1:** Read all images from related directories: Each image is in the 128x128x3 format. You should read and resize them to 64x64 format. Once you read the images, you have to convert them into vector format. Then it will become a vector with the format of 4096x1 size for each image. If there are n classes, for each class, you will have a matrix with 4096xn format. 

 **2. Stage2:** Select Eigenvalues and Eigenvectors of (PCA). From stage 1, you have obtained 4096xn matrices for each class. Now, we will apply PCA on extracted matrices. For this purpose, you have to implement process explained in the course note (08-PCA-Example.pdf).
   -PCA Step1: Extract covariance matrix for each class data (4096xn). Each covariance matrix will be nxn format.
   -PCA Step2: Extract eigenvalues and eigenvectors from covariance matrices. There will be n eigenvalues and n eigenvectors. Each eigenvector is in the nx1 format.
   -PCA Step3: Sort eigenvalues in descending order. Then select the eigenvectors corresponding to three (3) maximum eigenvalues. You will have a matrix containing nx3 eigenvectors
   
 **3. Stage3:** Extract PCA features by using Eigenvectors. In previous stage, you have extracted eigenvectors. Now, you will project the each class matrix (4096xn) into the eigenvector matrix of nx3 format. For this purpose, you have to multiply with eigenvector matrix. After projections, each class is represented with 4096x3 feature matrix. These  features are called as PCA features, or projections.
 
 **4. Stage4:** Test an Image In order to predict the class label of a test sample, you should follow below steps.
   -Step1: Read image from test directory
   -Step2: Convert into the test vector format 4096x1. Then, combine n times the test vector. The size will be 4096xn. Then, you have to multiply with eigenvector matrix of nx3. Each test sample is again represented with 4096x3 feature matrix.
   -Step3: Compute sum of absolute distance between train feature matrix and test feature matrix.
   
 ‼️‼️‼️ The label of test sample will be class name of corresponding minimum distance.

 ### HOMEWORK 4 : SVM Classification and Feature extraction
 #### Part 1:
The task involves conducting an experiment using the SVM (Support Vector Machine) classifier on the Caltech-101 dataset, which contains 15 classes with varying numbers of samples. The goal is to implement the one-against-all methodology to identify 15 different hyperplanes during the training stage, one for each class. In the testing stage, the SVM will project each test sample onto these hyperplanes to determine the class. The class associated with the highest similarity score is considered the best match, indicating the predicted target class for the sample.
 #### Part 2: Feature Extraction 
Read images in RGB format. Then convert image to LAB format. An image must be in the 224x224x3  format.  Instead of using the whole image data (224x224x3 size), we  have to extract some meaningful features in image. In this study, we will use HOG features. from skimage.color import rgb2lab https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog You can see that an image will be represented only with ndarray features. The dimension of feature vector per sample is 1xn. It means that training data (1457 samples) will be represented as (1457xn). Let’s call the X matrix as training matrix 
and y is label vector, which keeps the class name of samples. The size of X matrix is (1457xn) and the size of y vector is (1457x1). You are expected to fill the X matrix with features and y vector with class label per each sample. 
 #### Part 3: Train with SVM 
You are expected to train and test with SVM classifier. Create confusion matrix and show the accuracy for test samples. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html 
