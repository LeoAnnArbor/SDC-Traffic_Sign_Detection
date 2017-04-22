# Traffic Sign Classification

## Project goal and description

The goal of the project is to train a deep neural nework for Traffic Sign Recognition. Steps taken in this project include:

* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

---
## Explore, summarize and visualize the data set
Statistics of the traffic signs data set are as follows:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 

From visualization section (In[6]) in Traffic_Sign_Claasifier_submit.html, we notice that the original data are very biased, data for speed limit 50km/h has 2010 samples while data for speed limit 20km/h has only 180 samples. It's also worth noting that quality of the raw data varies drastically for the same traffic sign. Therefore, preprocessing of the raw data is frist performed. Data is first preprocessed via a pipline that sequentially perfomrs gamma correction, image sharpening and histogram equalization. As an example shown in Out[11] in Traffic_Sign_Claasifier_submit.html, image features are more clearly demonstaretd after preprocessing. To amend the bias issue, preprocessed images are then augmented via rotation for the cases with few data points. As illustrated in Out[16], after augmentation, number of images per class are roughly on the same order of magnitude. 

---
## Design, train and test a model architecture

My final model consisted of the following layers and the architecture is partially adopted from the paper "Multi-Column Deep Neural Network for Traffic Sign
Classification" :

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x100 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x150 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x150  				|
| Flatten				| outputs 1x5400        						|
| Fully connected		| outputs 1x300, dropout 0.5					|
| RELU					|												|
| Fully connected		| outputs 1x43									|
| Softmax				|												|
 
 To train the model, I used the following hyperparameters:
* epochs = 150
* batch size = 180
* learning rate = 0.0005
* initial weight is randomly assigned with zero mean and standard deviation of 0.01
 
The model os trained using AdamOptimizer to minimize the loss fuction, which is defined as cross entropy of the prediction. 

My final model results are:
* training set accuracy of 99.9%
* validation set accuracy of 99.8%
* test set accuracy of 96.8%

Throughout the training procedure, I made the following iterations primarily on hyperparmter tuning:
* In my First attempt, I used a training rate of 0.001 and  

---
## Test the model performance to make predictions on new images



The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

## Potential improvements 


