#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

I used the provided data set for track 1

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 80-90) 

The model includes RELU layers to introduce nonlinearity (code lines 81-85), and the data is normalized in the model using a Keras lambda layer (code line 80). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 27 we shuffle the samples before dividing it into training and test sets).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102). but the left/right camera params are tuned to keep the vehicle on track.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road applying a tuned parameter
to avoid swerving. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the provided data set with the Nvidida pipeline for self driving cars.

I thought this model might be appropriate because i tried myself to record training data, but the vehicle is not behaving good while training as i couldn't control the steering much even with the mousse.
Besides, i tried the LeNet Arch, but it was not giving promising results as LeNet is too simple i guess to create feature map for a 160x320 photo size.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it randolmy shuffles the data samples before creating the training and test sets, so that each patch in each Epoch can use different data for training.

Then I did too many iterations with the side cameras, without the side cameras, with image flipping, without, tuning the camera params, extending the training set with manually recorded new training data.
I also tried with LeNet pipeline for a while until i gave it up. it took me like 20 hours until i reached the model that i have now which is keeping the vehicle on track 1 for a complete lap.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specially the right swerve nd the swerve right after the bridge where no right hand lane marking was there.
 to improve the driving behavior in these cases, I tuned the side camera params, i cropped the images, resized the images and i used the data flipping.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 80-90) consisted of a convolution neural network with the following layers and layer sizes:
1- a Lambda layer to normalize the data (input data 40x80x3 images)
2- a conv layer witha filter size 5x5 and a depth of 24 with a relu activation function.
3- a conv layer witha filter size 5x5 and a depth of 36 with a relu activation function.
4- a conv layer witha filter size 5x5 and a depth of 48 with a relu activation function.
5- a conv layer witha filter size 3x3 and a depth of 64 with a relu activation function.
6- a conv layer witha filter size 3x3 and a depth of 64 with a relu activation function.
7- a layer to flatten the data to be ready for connecting the fully onnected layers.
8- a fully connected layer of size 576x100.
8- a fully connected layer of size 100x50.
8- a fully connected layer of size 50x10.
8- a final fully connected layer of size 10x1.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. example image of center lane driving you can find in the project resources provided t me as i used the provided data set.

To augment the data sat, I also flipped images and angles thinking that this would help passing the sharp right turn at the end of the track 1.

After the collection process, I had X number of data points. I then preprocessed this data by cropping the images, resizing the images and normalizing them with a Lambda layer.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the loss rates of the training and valid sets. I used an adam optimizer so that manually training the learning rate wasn't necessary.
