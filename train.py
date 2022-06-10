# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from CNN import CNN
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
	transforms.Resize(size=(112, 112)),
    transforms.ToTensor()
])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--mode", type=str, required=True,
	help="train/fine-tune")
ap.add_argument("--batch", type=int, required=False,
	help="batch size")
ap.add_argument("--eta", type=float, required=False,
	help="learning rate")
ap.add_argument("--epoch", type=int, required=False,
	help="learning rate")
ap.add_argument("--dataset", type=str, required=True,
	help="path to the dataset")
args = vars(ap.parse_args())
print(type(args["batch"]))
print(args["batch"])
# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 100

if args["batch"] is not None:
	BATCH_SIZE = args["batch"]
if args["eta"] is not None:
	INIT_LR = args["eta"]
if args["epoch"] is not None:
	EPOCHS = args["epoch"]


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] loading the dataset...")
dataset = ImageFolder(args["dataset"], transform=transform)
# calculate the train/validation split
print("[INFO] generating the train/validation split...")
# transform=ToTensor() in KMNIST, is it necessary, i guess no, it is for kmnist
(train, val, test) = random_split(dataset, [12000, 1500, 1500], generator=torch.Generator().manual_seed(42))


import torch.utils.data as data_utils

indices = torch.arange(3000)
train_subset = data_utils.Subset(train, indices)

indices = torch.arange(750)
val_subset = data_utils.Subset(val, indices)

test_subset = data_utils.Subset(test, indices)

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_subset, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(test_subset, batch_size=BATCH_SIZE)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

print("Dataset classes")
print(train.dataset.classes)
# initialize the CNN model
print("[INFO] initializing the CNN model...")
if args["mode"] == "train":
	model = CNN(
		numChannels=3,
		classes=len(train.dataset.classes), residual=True).to(device)
	# initialize our optimizer and loss function
	opt = Adam(model.parameters(), lr=INIT_LR)
	lossFn = nn.NLLLoss()
elif args["mode"] == "fine-tune":
	model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
	model = model.to(device)
	plist = [
			{'params': model.layer3.parameters(), 'lr': INIT_LR},
			{'params': model.layer4.parameters(), 'lr': INIT_LR}
			]

	# initialize our optimizer and loss function
	opt = Adam(plist, lr=INIT_LR)
	lossFn = nn.NLLLoss()
else:
	exit(1)

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()


# loop over our epochs
for e in range(0, EPOCHS):
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	# loop over the training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()

	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFn(pred, y)
			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()

    # calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect = valCorrect / len(valDataLoader.dataset)
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))

    # finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []
	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)
		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
# generate a classification report

targets = list()
for i in range(750):
	targets.append(test_subset.__getitem__(i)[-1])

targets = torch.tensor(targets)
preds = torch.tensor(preds)

print(classification_report(np.array(targets),
	np.array(preds), target_names=dataset.classes))

stacked = torch.stack((targets, preds), dim=1)
confusion_matrix = torch.zeros(15, 15, dtype=torch.int32)

# fill in the matrix
for row in stacked:
    true_label, pred_label = row.numpy()
    confusion_matrix[true_label, pred_label] += 1


from sklearn.metrics import confusion_matrix

# simply call the confusion_matrix function to build a confusion matrix
cm = confusion_matrix(targets, preds)
print(cm)

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, target_names, title='confusion matrix', cmap=None, normalize=False):
    if cmap is None:
        cmap = plt.get_cmap('Oranges')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylim(len(target_names)-0.5, -0.5)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig(title + '.png', dpi=500, bbox_inches = 'tight')
    
tt  = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

plot_confusion_matrix(cm, tt)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
# serialize the model to disk
torch.save(model, "trained_model.pt")

