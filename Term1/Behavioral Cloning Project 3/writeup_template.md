# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### The project requirements are listed here [rubric points](https://review.udacity.com/#!/rubrics/432/view) 
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

I have included the following
* [model.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/model.py) containing the script to create and train the model
* [drive.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/drive.py) for driving the car in autonomous mode
* [nvidia_cloning_model.h5](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/model/nvidia_cloning_model.h5) containing a trained convolution neural network for track one
* [nvidia_cloning_model_track2.h5](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/model/nvidia_cloning_model_track2.h5) containing a trained convolution neural network for track two
* [parameters.json](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/parameters.json) containing training and data augmentation parameters
* [DataAugmentation.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/DataAugmentation.py) implentation of data augmentation techniques for images
* [writeup_report.md](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/writeup_template.md])summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around:

- Track one by executing 
```sh
python drive.py nvidia_cloning_model.h5
```
-Track two using

```sh
python drive.py nvidia_cloning_model_track2.h5
```
#### 3. Submission code is usable and readable
[model.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/model.py) contains two convolutional neural networks, [LeNet](http://yann.lecun.com/exdb/lenet/) and [DAVE-2](https://arxiv.org/pdf/1604.07316.pdf). It includes the entire training pipeline from loading the data, augmentation techniques, training (either network or transfer learning from track one to track two) and validating the network.

The image preprocessing and augmentation is implemented in [DataAugmentation.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/DataAugmentation.py)

Training and data augmentation parameters are defined in [parameters.json](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/parameters.json)

This faciliated the training process on the AWS cloud instance as several changes and parameters had to be tuned.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

[DAVE-2](https://arxiv.org/pdf/164.07316.pdf) was implemented ([model.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Behavioral%20Cloning%20Project%203/model.py) lines 199-216)

The model includes 4 CNN layers with pooling layers in between, followed by four fully connected layers. The original model includes an additional normalisation layer which was implemented using a Keras lambda layer (code line 195). 

The model includes RELU layers to introduce nonlinearity (code line 208) and a tanh activation function.

I will not go into a discussion on LeNet has it has been implemented before, additionally it was'nt not used to generate the results used here.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated (20% split) on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The training and validation loss was monitored to avoid over training, by inspecting the parameter.json file it can be seen that the epochs were limited to 7 as the validation loss was stagnating.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 231).

Additionally, using a parameter (epochs, augmentation techinques, multiple datasets, networks...etc) file enabled rapid prototyping of the model on the AWS instance without having to inspect the entire code based which was a few hundred lines of code.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
