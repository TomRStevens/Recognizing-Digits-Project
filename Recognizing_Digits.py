import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageOps
import PIL
import tensorflow as tf
%matplotlib inline


## parameter denoting the batch size
BATCH_SIZE = 32

## defining a transformation that turns the data into Tensors
transform = transforms.Compose([transforms.ToTensor()])

## download and load training dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

## download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

## the MNIST set is built into PyTorch which is why we can just import it easily
## we can see that both the datasets have been imported and are ready for use
#print(trainset)
#print(testset)
## we will use trainloader and testloader to load the datasets once we need them


## defining the function to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

## get random images from the training set
dataiter = iter(trainloader)
for images, labels in dataiter:
    #print(images)
    break
#images, labels = dataiter.next()    # this didn't work so I am using the loop above instead

## show those images
imshow(torchvision.utils.make_grid(images))


## defining the model as a child class of nn.Module
## this makes it very easy to glue everything together
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        ## this hidden layer takes in the 28 * 28 image and returns 128 values (neurons)
        self.d1 = nn.Linear(28 * 28, 128)

        ## the dropout layer randomly zeroes some of the neurons in the neural network
        ## this is used as regularization and prevents overfitting
        self.dropout = nn.Dropout(p=0.2)

        ## takes the 128 values from the hidden layer and turns them into 10 outputs (one for each digit)
        self.d2 = nn.Linear(128, 10)

    ## the forward function is how data is fed into the model
    def forward(self, x):

        ## flatten the image and turns it from 2D into 1D
        x = x.flatten(start_dim = 1)

        ## feed the data through the hidden layer
        x = self.d1(x)

        ## optimization step
        x = F.relu(x)

        ## feed the data through the drop out layer
        x = self.dropout(x)

        ## feed the data through the output layer and get an output
        logits = self.d2(x)

        ## normalize the output into a probability distribution (percentages)
        out = F.softmax(logits, dim=1)

        ## get the final output
        return out


## learning rate...
learning_rate = 0.001

## number of times the network is trained on the data
num_epochs = 5

## determine which hardware the model will be trained on (gpu if possible)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## define model
model = MyModel()

## set the model to train on the correct hardware
model = model.to(device)

## this is a metric to compute the loss during training
criterion = nn.CrossEntropyLoss()

## optimization step that will be useful later
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


## defining the accuracy function
def get_accuracy(output, target, batch_size):
    ## looking at the amount of correct guesses
    corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()

    ## finding the average amount of correct guesses
    accuracy = 100.0 * corrects/batch_size

    ## output the accuracy
    return accuracy.item()


## train the model
for epoch in range(num_epochs):
    ## define variable that will be used to track the accuracy of the model throughout training
    train_running_loss = 0.0
    train_acc = 0.0

    ## commence training
    model = model.train()

    ## training step
    for i, (images, labels) in enumerate(trainloader):

        ## this will automatically give us images and their labels batches of 32 (or what we set the batch size to)
        images = images.to(device)
        labels = labels.to(device)

        ## predictions for that batch
        predictions = model(images)

        ## calculate loss
        loss = criterion(predictions, labels)

        ## reseting step
        optimizer.zero_grad()

        ## calculate the gradient of loss
        loss.backward()

        ## update model params
        optimizer.step()

        ## update the current loss and accuracy
        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(predictions, labels, BATCH_SIZE)

    model.eval()

    ## print the current loss and accuracy to update the user on the current status of training
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
          %(epoch, train_running_loss / i, train_acc/i)) 


## testing the model against the test dataset
test_acc = 0.0

## iterating through the test dataset
for i, (images, labels) in enumerate(testloader, 0):

    ## load the images and label in batches of 32 again
    images = images.to(device)
    labels = labels.to(device)

    ## get predictions
    outputs = model(images)

    ## calculate accuracy
    test_acc += get_accuracy(outputs, labels, BATCH_SIZE)

## print accuracy
print('Test Accuracy: %.2f'%( test_acc/i))


## defining a function to make predictions
def make_predictions(file_names):

    ## setting a transform to turn the images into tensors
    transform = transforms.Compose([transforms.ToTensor()])

    ## for each file in our file names list
    for a in file_names:

        ## setting the path to the digit
        path = f'data/handwritten/{a}'

        ## opening the image
        image = Image.open(path)

        ## resizing the image to the correct dimensions
        resized_image = image.resize((28, 28))

        ## turning the image into grayscale
        correct_image = ImageOps.grayscale(resized_image)

        ## the images are inverting so doing that here
        correct_image = PIL.ImageOps.invert(correct_image)

        ## turning the image into a tensor
        correct_image = transform(correct_image)

        ## start iterating through the test dataset
        for i,(images, labels) in enumerate(testloader, 0):

            ## setting the first image in the tensor to be that image (this is a very roundabout way I know)
            images[0] = correct_image

            ## set the images to the right hardware
            images = images.to(device)
            labels = labels.to(device)

            ## get predictions
            outputs = model(images)

            ## our current guess for what the digit is
            guess = 0

            ## the percentage chance the model thinks it is that digit
            num = 0

            ## for each possible digit
            for i in range(10):

                ## the percentage chance the model thinks it is that digit is given by the output
                percent = outputs[0][i]

                ## if that percent is higher than the one we currently have:
                if percent > num:

                    ## update the percent
                    num = percent

                    ## update the current guess
                    guess = i

                ## print each digit and the percentage chance the model gave it to be that digit
                print(f'{i} = {percent*100:.2f}%')

            ## print what the model predicted
            print(f"The model predicted {a} to be a {guess}")

            ## break out of the loop to only do this for one batch because I don't care about the rest of the test dataset
            break


file_names = ['7_1.jpeg','7_2.jpeg','7_3.jpeg','7_4.jpeg','7_5.jpeg','7_6.jpeg']
make_predictions(file_names)