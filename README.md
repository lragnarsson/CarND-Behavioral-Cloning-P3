# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup_images/parking_lot.png "Tricky parking lot"
[image2]: ./writeup_images/djungle_training.png "Djungle training"
[image3]: ./writeup_images/original_dataset.png "Original dataset"
[image4]: ./writeup_images/modified_dataset.png "Modified dataset"


## Rubric Points
[Source](https://review.udacity.com/#!/rubrics/432/view) 

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The model used had one convolutional layer with max-pooling followed by a fully connected layer.
Image cropping and normalization layers were added to the beginning using Keras lambda layers.

#### 2. Attempts to reduce overfitting in the model
The model contains a dropout layer in order to reduce overfitting (model.py lines 112). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 173).

#### 4. Appropriate training data
I trained my models on a combination of normal driving, recovery maneuvers and extra data for specific, tricky situations.
I augmented the data set by mirroring all images (and the labeled steering input).

## Model Architecture and Training Strategy
### 1. Solution Design Approach
I wanted to start by using transfer learning using one of the models [included](https://keras.io/applications/#documentation-for-individual-models) in Keras. Out of the models in that list, only a handful were available in the Keras vesion installed in the Udacity workspace. 

#### MobileNet
I wanted to try a smaller model, so I started with the 4 million parameter MobileNet model. It had the limitation that it required a square shaped input in order to use the imagenet weights. So I made an attempt to use the network architecture but to train the weights from scratch.

I replaced the input layer with one that fits the shape of an image from the center camera. Then added cropping and normalization to that. I excluded that last fully connected layer with the *include_top=False* argument. Then I flattened the output and added a fully connected layer going down to one dimension: the steering angle. I tried quite a few different things with the last fully connected layer. For example using activatity regularization to keep the driving behavior smooth, however I found out that the Keras version used had a [bug](https://stackoverflow.com/a/44496213) which made it not work. Another idea was to remove the bias term from the output layer - it should be equally likely that the car should turn left as right given a single image.

The code for the mobile net model is shown below

```python
# Transfer learning from https://keras.io/applications/#mobilenet
def mobile_net_model(input_shape=(160, 320, 3), y_crop=(50, 20)):
    # Define input layers with cropping and normalization:
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    input_tensor = Input(input_shape)
    # Crop image to remove car hood and background:
    cropping = Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape)(input_tensor)
    # Normalize image to zero mean even variance to better condition model training:
    normalizing = Lambda(lambda x: x/127.5 - 1., input_shape=new_shape, output_shape=new_shape)(cropping)
    # Load MobileNet model trained on imagenet
    base_model = MobileNet(dropout=1e-3, include_top=False, weights=None, input_tensor=input_tensor)
    # Add output layer:
    flat = Flatten()(base_model.output)
    out = Dense(1, use_bias=False)(flat)
    model = Model(base_model.input, out)
 
    return model
```

I only trained this model for a couple of epochs with a few different choices of hyper parameters, and it didn't show that much potential.
I wasn't able to load its parameters in the *drive.py* script either so I quickly abandoned this model.

#### InceptionV3
I then tried the 159 layer deep InceptionV3 network which I actually could use the imagenet trained weights for.
My hope was that I would only need to train the last fully connected layers in order to get a decent model. The results were OK.
I added code to lock the parameters of all except the last N layers, in order to try going back to more abstract features and training from there.

```python
# Transfer learning from https://keras.io/applications/#inceptionv3
def inception_v3_model(input_shape=(160, 320, 3), y_crop=(50, 20), layers_to_train=0):
    # Define input layers with cropping and normalization:
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    input_tensor = Input(input_shape)
    # Crop image to remove car hood and background:
    cropping = Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape)(input_tensor)
    # Normalize image to zero mean even variance to better condition model training:
    normalizing = Lambda(lambda x: x/127.5 - 1., input_shape=new_shape, output_shape=new_shape)(cropping)
    # Load InceptionV3 model trained on imagenet
    base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=input_shape) 

    # Disable training of parameters for every InceptionV3 layer except the last layers_to_train:
    for layer in base_model.layers:
        layer.trainable = False
        
    for layer in base_model.layers[-1 : -(1 + layers_to_train) : -1]:
        print(layer)
        layer.trainable = True
        
    # Add output layer:
    flat = Flatten()(base_model.output)
    out = Dense(1, use_bias=False)(flat)
    
    model = Model(base_model.input, out)
    return model
```

I tried re-training different numbers of layers in the network to see if there was a point where the layers trained on imagenet could be useful for steering the car.
In the end, I did not manage to train a network which could drive the entire course based on the InceptionV3 model. Because of the longer training times, I started looking at a simpler model instead.

#### Simple Shallow Convolutional Network
The third approach was to use a very simple convolutional network. It is loosly based on the Lenet architecture used in the previous project, but with fewer layers.
The idea was that the only thing we need to find is the edges of the road. Edges are one of the simplest type of features and is often what early convolutional layers in a CNNs learn to detect.

So a single convolutional layer with max pooling was followed by a fully connected layer down to 1 dimension was used for this model.

To combat overfitting, the model uses dropout after the max-pooling layer.

```python
# Hello Convolution World. This is the simplest model which was also able to be trained to complete one lap.
def simple_conv_model(input_shape=(160, 320, 3), y_crop=(55, 25)):
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    model = Sequential()
    # pre-processing layers:
    model.add(Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=new_shape, output_shape=new_shape))
    
    # Convolutional layer
    model.add(Conv2D(16, (5, 5), padding="valid", activation="tanh"))
    model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
    model.add(Dropout(0.5)) 
    
    # Fully connected -> steering angle output
    model.add(Flatten())
    model.add(Dense(1), use_bias=False)
    return model
```

#### Slightly More Complex Convolutional Network
The fourth and lat approach which was tested was to extends the simple convolutional network with one additional convolutional layer and two additional fully connected ones. However it did not show significant improvements over the simple network.

To combat overfitting, the model uses dropout after the first two fully connected layers.

```python
# Hello Convolution World. This is the simplest model which was also able to be trained to complete one lap.
def simple_conv_model(input_shape=(160, 320, 3), y_crop=(55, 25)):
    new_shape = (input_shape[0] - y_crop[0] - y_crop[1], input_shape[1], input_shape[2])
    model = Sequential()
    # pre-processing layers:
    model.add(Cropping2D(cropping=(y_crop, (0,0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=new_shape, output_shape=new_shape))
    
    # Convolutional layer
    model.add(Conv2D(16, (5, 5), padding="valid", activation="tanh"))
    model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
    model.add(Dropout(0.5)) 
    
    # Fully connected -> steering angle output
    model.add(Flatten())
    model.add(Dense(1), use_bias=False)
    return model
```


#### 2. Final Model Architecture

The final model architecture (model.py lines 106-122) consisted of a convolution neural network with the following layers

| Layer         		|     Description	        					            | 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 160x320x3 Front camera image			            		|
| Cropping         		| Producing a 80x320x3 image				            	|
| Normalization   		| Normalize image values to zero mean equal variance		|
| Convolution 5x5     	| 16 filters, 1x1 stride, valid padding                   	|
| tanh					| Activation function						            	|
| Max pooling	      	| 4x4 pool size and stride			                      	|
| Flatten   	      	|               	            		                  	|
| Fully connected		| Without bias, outputs steering angle, 1 dimension 		|


This simple model only needed a couple of epochs to converge with the adam optimizer. The resulting lap can be seen in **run1.mp4**.


#### 3. Creation of the Training Set & Training Process
I trained my models on a combination of normal driving, recovery maneuvers and extra data for specific, tricky situations.
With recovery data I mean frames where the car is almost about to leave the track and gives a large correctional steering input.
I did this because the trained networks performed OK until they got close to the edge and didn't seem to know what to do.

Early on I had problems with the car driving into the parking lot shown in the figure below.

![Parking lot][image1]

To avoid this, I added extra training data where I was driving past this parking lot, making sure to steer away from it. 
It was probably tricky because there was no obvious curb there. However, after adding the extra training data for this use case, it performed much better in this regard.

I also recorded a lap of driving on the jungle map in order to force the model to generalize better when it comes to colours and textures of the surroundings and of the road.

![Djungle training][image2]

I used a generator to create batches of training and validation data using sklearn's train_test_split function. The data was then shuffled.
I augmented the data set by mirroring all images (and the labeled steering input).
This was done inside the generator which means the mirrored version of an image was always in the same batch as the original.
I don't know if this had a positive or negative result on the training.

The main problem that still kept persisting was that the final model preferred driving straight when it was unsure of what to do. This was not helped by the fact that the training data set was heavily skewed towards straight line driving, see figure below.

![Original datasets][image3]

The different colors represent different subsets of the data which were all combined together for training and validation later on.

A quick and dirty fix for this was to remove much of the straight line data. This was done in model.py line 31.

```python
keep_prob = 0.4 + 10 * abs(float(line[3])) # Remove much of straight-ish line driving
if random.random() < keep_prob: 
    data_samples.append(line)
```
The resulting distribution can be seen below.

![Modified datasets][image4]

This made the model much more eager to turn, which helped it in the sharp corners. The negative side effect was a more snake-like driving style. After doing this the simple model was able to be trained to drive around the track without leaving the road. However it still came close to the edge a few times without adequate reaction. In order to improve this the dataset was further augmented to use the left and right cameras with a steering angle correction of +- 0.25 compared to the actual steering angle. The idea behind this was to get more training data where the car is steering away from the edge of the road. This data augmentation method greatly improved the performance of trained models.