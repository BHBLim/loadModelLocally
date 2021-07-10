import torch.nn as nn 
import torch 
from torchvision import transforms,models,datasets 
from PIL import Image 
import numpy as np 
from torch import optim 

transform = transforms.Compose([transforms.Resize(255), 
                                transforms.CenterCrop(224), 
                                transforms.ToTensor()]) 

#You can change the image path here.
IMAGE_PATH = 'images/dog.jpg'
#If you want, you can modify this script to accept command line arguments instead of coding
#it here :D Google is your friend

#Put your model here! I haven't uploaded it to GitHub because generally you shouldn't
#check-in large files.
MODEL_PATH = "models/dogscats.pt"

#Load the model structure of densenet121
model = models.densenet121(pretrained = False)
for params in model.parameters(): 
    params.requires_grad = False 

from collections import OrderedDict 

#Define our model's own last layer which we used earlier in the Google Colab file
classifier = nn.Sequential(OrderedDict([ 
    ('fc1',nn.Linear(1024,500)), 
    ('relu',nn.ReLU()), 
    ('fc2',nn.Linear(500,2)), 
    ('Output',nn.LogSoftmax(dim=1)) 
]))

#Modify the classifier layer of the densenet121 to ours
model.classifier = classifier 

#We have to specify that we're using CPU since we previously trained on GPU
#We load our saved parameters
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
#Re-evaluate the model
model.eval()

# Open image
image = Image.open(IMAGE_PATH).convert('RGB')

#Convert the input into a format the model expects
input = transform(image)
input_batch = input.unsqueeze(0) #The unsqueeze part is a bit tricky, it's done in Colab when loading the dataset

#Get the probability of each output class
output = model(input_batch) 

#Select the highest probabilty as the output class
pred = torch.argmax(output, dim=1) 

#Convert this pytorch tensor into a normal Python number
pred = [p.item() for p in pred]

#Map the numerical class to a label
#use dataset.class_to_idx on the dataset on the Google Colab file to see the mapping of index to class
labels = {0:' This izza kitty', 1:' This is doge'}
print(labels[pred[0]])