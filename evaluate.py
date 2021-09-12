import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np 
import argparse
import os 

import matplotlib.pyplot as plt

from utils.utils import check_path, get_logger, dataset_split, count_classes
from dataloader import RakutenCatalogueLoader
import torchvision
from torchvision.transforms import Compose, ColorJitter, RandomHorizontalFlip, Normalize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description = 'PyTorch evaluation of MobileNetV2 on Rakuten Catalog data')
parser.add_argument('--checkpoint', default = 'checkpoints/final_model.pt', type = str, help = 'Checkpoint to evaluate')
parser.add_argument('--num_workers', default = 8, type = int)
parser.add_argument('--seed', default = 42, type = int)
parser.add_argument('--test_file', default = 'data/test_data.csv', type = str)

# Parse arguments
args = parser.parse_args()

# Logger
logger = get_logger()

def main():
	
	# For reproducibility
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)

	# For faster computation (if possible)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('==> Currently running on {}'.format(device))

	# Define the model 
	model = mobilenet_v2()


	# We need to adapt the last layer of the classifier head 
	# For the right number of classes in the dataset 
	# We load the checkpoint we want to use for the prediction
	logger.info('==> Loading the desired checkpoint for inference')
	checkpoint = torch.load(args.checkpoint)
	model.classifier[1] = nn.Linear(1280, list(checkpoint['model_state_dict']['classifier.1.bias'].shape)[0])
	model.load_state_dict(checkpoint['model_state_dict'])
	
	model.to(device)
	model.eval()

	val_transform_list = [ToTensor()]
	val_transform = Compose(val_transform_list)

	# Test loader 
	test_data = RakutenCatalogueLoader(args.test_file, 'data/images', transform = val_transform)
	logger.info('==> {} testing samples found in the testing set'.format(len(test_data)))

	# Test loader
	test_loader = DataLoader(test_data, batch_size=1,
                        shuffle=False, num_workers=args.num_workers)


	logger.info('==> Testing the model')

	acc = 0
	for i, data in enumerate(test_loader):

		# Inputs and GT
		image, target = data['image'], data['target']

		# Move them to gpu 
		image = image.to(device)
		target = target.to(device)

		# Predict
		pred_target = model(image)

		y_pred_softmax = torch.softmax(pred_target, dim = 1)
		_, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

		acc += torch.sum(target== y_pred_tags) 

	logger.info('==> Test accuracy: {}%'.format(round((acc.item()/len(test_loader))*100,3)))


if __name__ == '__main__':
	main()