import numpy as np
import pandas as pd
from PIL import Image
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import sys
import csv
import random

# Dataset subclass for loading images efficiently in memory
class WhalesDataset(data.dataset.Dataset):
  def __init__(self, is_train, transform):
    self.is_train = is_train
    self.transform = transform
  def __len__(self):
    if self.is_train:
      return train_csv.shape[0]  # training images
    return 7960 # testing images

  def __getitem__(self, index):
    if self.is_train:
      whale_id = train_labels_map[train_csv.iloc[index]['Id']] # [] is loc so use iloc!!!
      img_path = train_csv.iloc[index]['Path']
      img = Image.open(img_path)
      if self.transform:
        img = self.transform(img) # ToTensor converts (HxWxC) -> (CxHxW)
      return index, whale_id, img
    else:
      img_path = os.listdir('./data/test')[index]
      img = Image.open('./data/test/' + img_path)
      if self.transform:
        img = self.transform(img) # ToTensor converts (HxWxC) -> (CxHxW)
      return index, img_path, img

def train():
  num_classes = len(train_labels)
  # Hyperparameters
  num_epochs = 75
  first_learning_rate = 0.000075
  second_learning_rate = 0.00005
  third_learning_rate = 0.000025
  fourth_learning_rate = 0.00001
  train_params = {'batch_size': 20, 'shuffle': True, 'num_workers': 5}
  test_params = {'batch_size': 20, 'shuffle': True, 'num_workers': 5}
  train_valid_params = {'batch_size': 40, 'shuffle': True, 'num_workers': 5}

  # Load Data
  train_steps = transforms.Compose([
    transforms.RandomResizedCrop(size=256),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(30),
    transforms.ColorJitter(brightness=0.3),
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(224), # ImageNet standard
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
  ])

  test_steps = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(224), # to handle images larger, have to modify last layer's input features
    transforms.ToTensor(),
    # Necessary for pre-trained models ...
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
  ])

  # Only convert to tensor (for test time augmentation)
  test_load = transforms.Compose([
    transforms.Resize(256),
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(256),
    transforms.ToTensor()
  ])
    
  train_set = WhalesDataset(is_train = True, transform=train_steps)
  train_valid_set = WhalesDataset(is_train = True, transform=test_load)
  test_set = WhalesDataset(is_train = False, transform=test_load)
  train_loader = data.DataLoader(train_set, **train_params)
  train_valid_loader = data.DataLoader(train_valid_set, **train_valid_params)
  test_loader = data.DataLoader(test_set, **test_params)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = torchvision.models.resnet50(pretrained=True).to(device)
  #model = torchvision.models.resnet101(pretrained=True).to(device)
  model = nn.DataParallel(model) # enable parallelism
  # Freeze all layers
  for i, param in model.named_parameters():
    param.requires_grad = False
  # ImageNet has 1000 classes, so we need to change last layer to accomodate the number of classes we have
  # If wrapped in DataParallel, use .module.fc instead of .fc
  imagenet_features = model.module.fc.in_features
  model.module.fc = nn.Sequential(nn.BatchNorm1d(imagenet_features), nn.Dropout(p=0.5), nn.Linear(imagenet_features, num_classes))
    
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=first_learning_rate)
    
  # Train the network
  total_steps = len(train_loader)
  iterations = []
  losses = []
  for epoch in range(num_epochs):
    # Train whole model
    if epoch == 3:
      for i, param in model.named_parameters():
        param.requires_grad = True
      optimizer = torch.optim.Adam(model.parameters(), lr=first_learning_rate)
    if epoch == 15:
      optimizer = torch.optim.Adam(model.parameters(), lr=second_learning_rate)
    if epoch == 30:
      optimizer = torch.optim.Adam(model.parameters(), lr=third_learning_rate)
    if epoch == 45:
      optimizer = torch.optim.Adam(model.parameters(), lr=fourth_learning_rate)
    for i, (_, whale_ids, images) in enumerate(train_loader):
      whale_ids = torch.tensor(np.array(whale_ids)).to(device)
      whale_ids.cuda()
      # Send tensors to GPU
      # whale_ids = whale_ids.to(device) # batch_size x 1
      images = images.to(device)   # batch_size x 3 x 224 x 224
      model.cuda()
      images.cuda()
      model.train() # reset model to training mode

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, whale_ids)
      # use backward() to do backprop on loss variable
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % 50 == 0:
        curr_iter = epoch * len(train_loader) + i
        iterations.append(curr_iter)
        losses.append(loss.item())
        print ('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_steps, loss.item()))
        sys.stdout.flush()
    if (epoch + 1) % 5 == 0:
      curr_acc = train_acc(model, train_valid_loader, test_steps, num_iters = 5)
      print('Approx. training accuracy: {}'.format(curr_acc))
  
  # Calculate loss over larger subset of training set instead of batch
  final_acc = train_acc(model, train_valid_loader, test_steps, num_iters = 20)
  print('Final training subset accuracy: {}'.format(final_acc))
  print('Making predictions on testing set:')
  make_predictions(model, test_loader, test_steps)

def map_for_image(label, predictions):
  """Computes the precision score of one image.
  Parameters
  ----------
  label : string (The true label of the image)
  predictions : list (of up to 5 predicted elements - order matters)
  Returns
  -------
  score : double
  """
  try:
    return 1 / (predictions[:5].index(label) + 1)
  except ValueError:
    return 0.0

# Return augmented images for TTA at validation/prediction time
def augment_images(images, test_steps = None):
  augmentation = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(30),
    transforms.ColorJitter(brightness=0.3),
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
  ])
  if test_steps is not None:
    augmentation = test_steps
  new_images = torch.zeros([images.shape[0], 3, 224, 224])
  for i in range(0, images.shape[0]):
    new_images[i] = augmentation(images[i])
  return new_images

def train_acc(model, loader, test_steps, num_iters):
  # Test the mdoel
  model.eval()
  
  acc = 0.0
  iters = 0
  with torch.no_grad():
    for _, whale_ids, images_orig in loader:
      # Just cropping
      images1 = augment_images(images_orig, test_steps)
      # Three augmentations
      images2 = augment_images(images_orig)
      images3 = augment_images(images_orig)
      images4 = augment_images(images_orig)
      images1 = images1.cuda()
      images2 = images2.cuda()
      images3 = images3.cuda()
      images4 = augment_images(images_orig)
      batch_size = len(whale_ids)
      whale_ids = torch.tensor(np.array(whale_ids)).cuda()
      model.cuda()

      # Original unaugmented image (except cropped)
      output1 = model(images1)
      output2 = model(images2)
      output3 = model(images3)
      output4 = model(images4)
      outputs = (output1 + output2 + output3) / 4
      outputs = nn.functional.softmax(outputs, 1)
      probs, predicted = torch.topk(outputs.data, 5, dim=1) # batch_size x 5
      
      for i in range(predicted.shape[0]):
        true_label = train_labels[whale_ids[i]]
        predicted_labels = [train_labels[predicted[i, j]] for j in range(5)]
        for pred_idx in range(5):
            if probs[i, pred_idx] < new_whale_thresh and 'new_whale' not in predicted_labels:
              predicted_labels.insert(pred_idx, 'new_whale')
              predicted_labels = predicted_labels[:5]
        acc += map_for_image(true_label, predicted_labels)
      iters += 1
      if iters == num_iters:
        model.train() # reset model 
        return acc / batch_size / iters
    model.train() # reset model
    return acc / batch_size / iters

def make_predictions(model, loader, test_steps):
  model.eval() # eval mode
  with torch.no_grad():
    results = []
    seenImg = {}
    for _, image_paths, images_orig in loader:
      # Just cropping
      images1 = augment_images(images_orig, test_steps)
      # Two augmentations
      images2 = augment_images(images_orig)
      images3 = augment_images(images_orig)
      images4 = augment_images(images_orig)
      images1 = images1.cuda()
      images2 = images2.cuda()
      images3 = images3.cuda()
      images4 = images4.cuda()
      model.cuda()

      # Original unaugmented image (except cropped)
      output1 = model(images1)
      output2 = model(images2)
      output3 = model(images3)
      output4 = model(images4)
      outputs = (output1 + output2 + output3 + output4) / 4
      outputs = nn.functional.softmax(outputs, 1)
      probs, predicted = torch.topk(outputs.data, 5, dim=1) # batch_size x 5
     
      for i in range(predicted.shape[0]):
        predicted_labels = [train_labels[predicted[i, j]] for j in range(5)]
        for pred_idx in range(5):
          if probs[i, pred_idx] < new_whale_thresh and 'new_whale' not in predicted_labels:
            predicted_labels.insert(pred_idx, 'new_whale')        
            predicted_labels = predicted_labels[:5]
        # Store csv rows
        if image_paths[i] not in seenImg:
          row = [image_paths[i]]
          row.append(" ".join(predicted_labels))
          results.append(row)
          seenImg[image_paths[i]] = True
    model.train() # reset
    results.sort() # sort by filename (the first substring of each row)
    results.insert(0, ['Image', 'Id'])
    with open(test_csv_path, 'w') as predictionFile:
      wr = csv.writer(predictionFile)
      wr.writerows(results)

if __name__ == '__main__':
  # Set paths
  train_path = os.path.abspath('./data/train')
  test_path = os.path.abspath('./data/test')
  csv_path = os.path.abspath('./data/train.csv')
  test_csv_path = os.path.abspath('./predictions.csv')

  new_whale_thresh = 0.38
  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.229, 0.224, 0.225]

  # Read train.csv into Pandas dataframe
  train_csv = pd.read_csv(csv_path)
  # Remove new whales
  train_csv = train_csv[train_csv['Id'] != 'new_whale']
  # Add absolute image path to make reading easier
  train_csv['Path'] = [os.path.join(train_path, img) for img in train_csv['Image']]
  train_labels = train_csv['Id'].unique()
  train_labels_map = {train_labels[i] : i for i in range(0, len(train_labels))}
  train()
