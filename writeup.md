#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[image1]:./Imagenes_escaladas/100_1607.jpg "Yield"
[image2]:./Imagenes_escaladas/5155701-German-traffic-sign-No-205-give-way-Stock-Photo.jpg ""Right of way at the next intersection"
[image3]:./Imagenes_escaladas/no-entry.jpg "No entry"
[image4]:./Imagenes_escaladas/roundabout.jpg "Mandatory roundabout"
[image5]:./Imagenes_escaladas/speedlimit30.jpg "Speed limit of 30"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy and the pickle library for loading the data as you can check here:
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
With "matplotlip.pyplot" I can have a graphical visualization of the data:
plt.imshow(image)

With "random" I can randomize the data:
index = random.randint(0, len(X_train) )



####2. Include an exploratory visualization of the dataset.
With the "numpy" library I get the maximum and the minimum value that pixels can have:
max_pix_val = np.max(np.max(X_train[:,:,:,:],axis=(1,2,3)))
min_pix_val = np.min(np.min(X_train[:,:,:,:],axis=(1,2,3))

Here I measure the length and the size of the pictures from valid, test and train.p
n_train = len(X_train)
n_validation = len(X_validation)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(set(y_train))



###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The first thing I do for preprocessing my images is making an histogram equalization. This process increases the global contrast of my images, especially when the usable data of the image is represented by close contrast values. This method is useful with images with backgrounds and foregrounds that are both bright or both dark.
It produces unrealistic effects in photographs although is very useful for analyzing images. 
The only parameter I tuned was "clip_limit" that allows me to adjusts how much the contrast is going to change. 

First I need to convert my images from rgb to hsv:
X_train_new = [x/255 for x in X_train]
X_train_hsv = matplotlib.colors.rgb_to_hsv(X_train_new)

Once I do that, I have to apply it to the "v" layer and join all the layers alltogether again: 
X_train_h, X_train_s, X_train_v = X_train_hsv[:,:,0], X_train_hsv[:,:,1], X_train_hsv[:,:,2]
X_train_eq = skimage.exposure.equalize_adapthist(X_train_v, kernel_size=None, clip_limit=0.05, nbins=256 )
X_train_hsv[:,:,2] = X_train_eq

The I pass everything to rgb as I started:
X_train = matplotlib.colors.hsv_to_rgb(X_train_hsv)

Now it's time for normalization. If we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another. We might be over compensating a correction in one weight dimension while undercompensating in another.

This is non-ideal as we might find ourselves in a oscillating (unable to center onto a better maxima in cost(weights) space) state or in a slow moving (traveling too slow to get to a better maxima) state. I have defined a function that computes the mean and the standard deviation. I need a mean of zero for comparing all the images between eachother:
def normalize(arr, mean = None, std = None):
    if mean is None:
        mean = np.mean(arr, axis = 0)
    if std is None:
        std = np.std(arr, axis = 0)
    new_arr = (arr-mean)/std

    return new_arr, mean, std
    
I haven't added any new data.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|It eliminates negative values                  |
  Dropout 			It eliminiates random outputs for priventing overfitting.
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 2x2 stride, same padding, outputs 5x5x16		|
| Fully connected		| input 84, output 10    					|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used cross entropy that tells me how different is my data, loss_operation that calculates the reduced mean. The optimizer I used is the AdamOptimizer that asks me for the "learning rate". I've set this learning rate as 0.001:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

The size of my EPOCHS is of 100 and the batch_size 128. The mean from my set is 0 and the standard deviation is 0.1.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.937
* validation set accuracy of 0.991
* test set accuracy of 0.929

I used the Lenet architecture that was taught during this course. The Lenet architecture it is small and straightforward what makes it perfect for learning the basics. 
At first I started with the standard Lenet arch. but I detected that with that number of EPOCHS it never reached a good accuracy so I decided to increase it.
After doing that I saw that commonly, training acc. reached way higher values than validation accuracy so I thought that this was due to overfitting.
For solving this problem I used the technique called "dropout" that consists in blocking some of the outputs before them arrive to the next layers.
Another parameter that was cruzial for obtaining that accuracy was clip_limit inside the equalization. First I tried with 0.01 and 0.05 and then I realised that
with 0.03 I get the best performance.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The 30km/h speed limit signal can be difficult to calssify because is coloured with red color in the borders so it can be confused with a prohibition signal. Usually my model doesn't have problems with the rest of the signals.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way     		| Right of way   								| 
| Right of way at the next intersection | Right of way at the next intersection |
| Yield					| Yield											|
| 30 km/h	      		| Limit 60km/h
| Mandatory roundabout	| Mandatory roundabout 							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
The limit of 30Km/h signal is problematic because all the speed limit signals are quite similar and the model tends to confuse them.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way     		| 0.98								| 
| Right of way at the next intersection | 0.98 |
| Yield					| 0.99											|
| 30 km/h	      		| 0.55
| Mandatory roundabout	| 0.7 							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


