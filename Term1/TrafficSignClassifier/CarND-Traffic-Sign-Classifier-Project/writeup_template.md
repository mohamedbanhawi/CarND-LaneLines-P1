# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### 1. README 

This is a read me for my [project code](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used Numpy to calculate the statistics of the data set since the data was provided as [ndarray](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html):

Each dataset contains equal number images and corresponding labels

* The size of training set is: 34799 
* The size of the validation set is: 
* The size of test set is: 34799
* The shape of a traffic sign image is: (32, 32, 3) such that each image is 32x32 pixels and 3 channels (RGB)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I used a histogram to visualise the provided datasets, it is clear that the data is not uniformly distributed across the classes.

**Training**
![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/init_train.png)

**Validation**
![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/init_valid.png)

**Testing**
![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/init_tain.png)

I found it useful to plot the different class to identify any potential differences that would be helpful in desining the network.

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/all_classes.png)

### Design and Test a Model Architecture

#### 1. Preprocessing the image data. 

##### a. Augmented "Jittered" Dataset

The work in [1] provided insight into the preprocessing stage for CNN. Typically, CNN are invariant to geometric transformation in the image, as such "jittering" the image would make the model more robust. The jittered dataset were randomly perturbed in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees).

Based on that concept of jittering the dataset, I used 5 different processes to created a jittered dataset. OpenCV had built in functionality to perform the those transformation.

1. Rotation

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/example_rotate.png)

2. Translation

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/example_translate.png)

3. Blurring

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/example_blur.png)

4. Gamma Correction

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/example_gamma.png)

##### b. Unifiying the dataset classes

The dataset distribution appears to be non uniform. 

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/post_process_all.png)

To avoid the network overfitting a particular class, I artificially added datapoints by randomly applying one of four "jittering" functions for undersampled classes (random parameters and functions).

The final training dataset distribution 

![alt text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/uniform.png)

##### c. Zero Centering

Zero centring appears to be a standard process. The data was converted to grayscale and the normalised and centred. This process appears to improve the gradient descent search.

#### 2.The model architecture 

My final model consisted of the following layers was rather similar to the LeNet5 [2] architecture
I added dropout layers to avoid relying on a certain class for training since the data was not uniform across all classes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| 1. Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x16 	|
| RELU					|		To introduce nonlinearlity										|
| 2. Convolution 3x3	    | 1x1 stride, valid padding, outputs 28x28x32. |
| RELU					|		To introduce nonlinearlity										|
| 3. Max pooling	      	| 2x2 stride, 2x2 Kernel  outputs 16x16x64 				|
| Drop out | Keep probability 0.6 |
| 4. Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x64. |
| RELU					|		To introduce nonlinearlity										|
| 5. Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x64. |
| RELU					|		To introduce nonlinearlity										|
| 6. Max pooling	      	| 2x2 stride, 2x2 Kernel  outputs 5x5x64. 				|
| 7. Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x128. |
| RELU					|		To introduce nonlinearlity										|
| Drop out | Keep probability 0.6 |
| 9. Fully connected		| Input = 1152, Output 1024        									|
| 10. Fully connected		| Input = 1024, Output 43 Classes        									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used varied the learing rate, batch size and epochs.

Batch size seemed to affect the speed of the training process.
However, it seemed like the model converged around 95%.

Epochs were limited to 15 to avoid over training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.7%
* test set accuracy of 93.4 %

I added drop out to avoid relying on a particular datapoint.

If a well known architecture was chosen:
* What architecture was chosen?
I modified the LeNet5 [2] to match the approach in the paper [1].

* Why did you believe it would be relevant to the traffic sign application?
I estimated that at least 3 Convolutional layers would be required to classify the frame of the sign. Additional three would be required to classify the inner shape of the sign.

I added more fully connected layers as I suspected this would be needed to classify all the different classes.

Finally, ReLu were selected as the classification at hand is highly nonlinear.

Several additions were not added due to time limtations, the authors in [1] suggested using YUB colorspace. Additionally, other architectures such as GoogLeNet [3] 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: 

[Online Dataset](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/online_test)

I suspected that the ahead image would be difficult to classify due to the shadow of another pole that might appear like a left arrow which is another class in the dataset.

![alt_text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/ahead.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead     		| ![alt_text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/classified.png)  									| 
| Speed Limit 100 Km/h     		| ![alt_text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/classified1.png)  									| 
| Road work      		| ![alt_text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/classified2.png)  									| 
| Speed Limit 80 Km/h     		| ![alt_text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/classified3.png)  									| 
| Roundabout Mandatory      		| ![alt_text](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/TrafficSignClassifier/CarND-Traffic-Sign-Classifier-Project/write_up_images/classified4.png)  									| 

The model accuracy is 100%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

I used to calculate the top 5 soft max probabilites for each image.

The code block is shown below,
```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    y_predict = sess.run(tf.nn.softmax(logits), feed_dict={x: X_online})
    top_signs = sess.run(tf.nn.top_k(y_predict, k =5))
```


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

#### References
[1] Sermanet, Pierre, and Yann LeCun. "Traffic sign recognition with multi-scale convolutional networks." Neural Networks (IJCNN), The 2011 International Joint Conference on. IEEE, 2011.

[2] LeCun, Yann. "LeNet-5, convolutional neural networks." URL: http://yann. lecun. com/exdb/lenet (2015): 20.
