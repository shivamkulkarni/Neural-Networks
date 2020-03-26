# Convolutional Neural Network
## Dataset
Name - Fashion mnist<br />
Number of Images - 70000<br />
Image Size - 28 x 28 <br />
Number of Labels - 10<br />

## Model Summary<br />
Library- Keras<br />
Model - CNN<br />
Number of Epochs - 10<br />
Activation Function for Convulutional Layer- ReLU<br />
Activation Function for Output Layer- Softmax<br />
Loss function - categorical_crossentropy<br />
Optimizer - Adam<br /><br />

## Layers
| Layer (type)                 	|    Output Shape    	| Param # 	|
|------------------------------	|:------------------:	|---------	|
| conv2d_1 (Conv2D)            	| (None, 26, 26, 32) 	| 320     	|
| max_pooling2d_1 (MaxPooling2 	| (None, 13, 13, 32) 	| 0       	|
| flatten_1 (Flatten)          	|    (None, 5408)    	| 0       	|
| dense_1 (Dense)              	|     (None, 100)    	| 540900  	|
| dense_2 (Dense)              	|     (None, 10)     	| 1010    	|

## Run Summary<br />
accuracy - 0.94<br />
loss - 0.14<br />
val_accuracy - 0.88<br />
val_loss - 0.44<br />
runtime - 614.183<br /><br />
