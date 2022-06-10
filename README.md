To run this program:

```python train.py --mode=train --batch=32 --eta=0.001 --epoch=100 --dataset="./pa3_dataset"```
```python train.py --mode=fine-tune --batch=32 --eta=0.001 --epoch=100 --dataset="./pa3_dataset"```

The two different models come under the same CNN class with a subtle difference in using parameters.
Filter size is 3 by 3 for the models and the stride is set to 1. Each model is composed of 5 convolutional
layers and a fully connected layer. The first convolutional layer takes 3 channeled input and outputs
6 channels. The second convolutional layer takes 6 channels and outputs 6 channels. The third
convolutional layer takes 6 channels and outputs 16 channels. The fourth convolutional layer takes 16
channels and outputs 16 channels. The fifth and last convolutional layer takes 16 channels and outputs
16 channels. The last layer of the CNN is a fully connected layer whose dimension 102*102*16.
To be used in the residual connections, there is an extra block that simply fits the size of the input
to the residual input to make them compatible.