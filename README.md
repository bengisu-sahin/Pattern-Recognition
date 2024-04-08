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
6. **Output **:
   Output for test code is like this
![image](https://github.com/bengisu-sahin/Pattern-Recognition/assets/71591780/8cb3b37c-e176-44bd-b788-09b2b4987644)
