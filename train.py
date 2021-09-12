import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np 
import argparse
import os 

import matplotlib.pyplot as plt

from utils.utils import check_path, get_logger, dataset_split, count_classes, clean_dataset
from dataloader import RakutenCatalogueLoader
import torchvision
from torchvision.transforms import Compose, ColorJitter, RandomHorizontalFlip, Normalize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description = 'PyTorch training of MobileNetV2 on Rakuten Catalog data')

# Training data
parser.add_argument('--data_dir', type = str, default='data/images')
parser.add_argument('--csv_file', type = str, default='data/data_set.csv')
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--num_workers', type = int, default = 8)

# Model 
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--checkpoint_dir', type=str, default = 'checkpoints',
						  help='Directory to save model checkpoints and logs')
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
parser.add_argument('--checkpoint', default = None, type = str, help = 'Checkpoint from which resume training')

# Log
parser.add_argument('--save_ckpt_freq', default=5, type=int, help='Save checkpoint frequency (epochs)')
parser.add_argument('--tensorboard_display', action='store_true', help = 'Enable tensorboard logging')

# Parse arguments
args = parser.parse_args()

if args.tensorboard_display:
	print('Tensorboard display enabled.\nRun tensorboard --logdir=runs')
	writer = SummaryWriter()

# Generated corrected dataset csv
# Because some images in the image folder are not in the csv file 
if not os.path.isfile('data/corrected_data_set.csv'):
	clean_dataset(args.csv_file)

# Check if checkpoint path exists
check_path(args.checkpoint_dir)

# Logger (for logging :D)
logger = get_logger()

def main():
	
	# For reproducibility
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	 
	 # For faster computation (if possible)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info('==> Currently running on {}'.format(device))

	 # A few data augmentations 
	train_transform_list = [#ColorJitter(0.2,0.2,0.2,0.2),
							#RandomHorizontalFlip(),
							ToTensor()
							]

	val_transform_list = [ToTensor()]

	# Compose the transformer 
	train_transform = Compose(train_transform_list)
	val_transform = Compose(val_transform_list)

	# Split the data into train, val, and test
	# We'll use a 90/5/5 split, as the task is quite hard 
	# And the dataset is relatively small 
	logger.info('==> Splitting data into train, val, test')
	train_csv, validate_csv, test_csv = dataset_split('data/corrected_data_set.csv')
	classes, nb_classes = count_classes(args.csv_file)

	logger.info('==> Number of classes : {}'.format(nb_classes))

	# Build a dictionnary to deal with the classes 
	cl_dict = {}
	for i,cl in enumerate(classes):
		cl_dict[cl] = i

	print(cl_dict)

	# Build the dataloaders
	train_data = RakutenCatalogueLoader(train_csv, args.data_dir, transform = train_transform)
	logger.info('==> {} training samples found in the training set'.format(len(train_data)))
	
	val_data = RakutenCatalogueLoader(validate_csv, args.data_dir, transform = val_transform)
	logger.info('==> {} validation samples found in the validation set'.format(len(val_data)))


	# Training loader 
	train_loader = DataLoader(train_data, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)
	# Validation loader
	val_loader = DataLoader(val_data, batch_size=1,
                        shuffle=False, num_workers=args.num_workers)

	# Define the model 
	model = mobilenet_v2()
	# Define the optimizer 
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params  = model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)

	# We need to adapt the last layer of the classifier head 
	# For the right number of classes in the dataset 
	model.classifier[1] = nn.Linear(1280, nb_classes)

	# If the choice is to resume the training 
	if args.resume : 

		logger.info('==> Loading Checkpoint for Resuming Training')
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.to(device)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		iterations = checkpoint['iterations']

	# Otherwise init and begin training
	else : 

		start_epoch = 0
		iterations = 0
		model.to(device)
		# Training the model 
		logger.info('==> Started Training')



	for epoch in range(start_epoch, args.max_epoch):

		logger.info('==> Training Epoch {}'.format(epoch))
		model.train()
		running_loss = 0.0

		for i, data in enumerate(train_loader):
			iterations += 1
			image, target = data['image'], data['target']

			
			image = image.to(device)
			target = target.to(device)

			optimizer.zero_grad()

			pred_target = model(image)

			loss = criterion(pred_target, target)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if args.tensorboard_display : 
				writer.add_scalar("Loss/train", loss, iterations)

			if i % 100 == 99:    # print every 50 mini-batches
				logger.info('==> [%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 100))
			
				grid = torchvision.utils.make_grid(image)

				if args.tensorboard_display : 
					writer.add_image('images', grid, 0)
					writer.add_graph(model, image)

				running_loss = 0.0		

		logger.info('==> Validation for Epoch {}'.format(epoch))
		model.eval()
		acc = 0
		for i, data in enumerate(val_loader):

			image, target = data['image'], data['target']

			image = image.to(device)
			target = target.to(device)

			pred_target = model(image)

			y_pred_softmax = torch.softmax(pred_target, dim = 1)
			_, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

			acc += torch.sum(target== y_pred_tags) 

		logger.info('==> [%d, _] Validation accuracy: %.3f' %
					(epoch + 1, acc.item()/len(val_loader)))
		if args.tensorboard_display : 
			writer.add_scalar("Accuracy/val", acc.item()/len(val_loader), epoch + 1)

		if epoch % args.save_ckpt_freq == args.save_ckpt_freq - 1 :
			logger.info('==> Saving checkpoint at Epoch {}'.format(epoch))
			torch.save({
            	'epoch': epoch,
            	'iterations':iterations,
            	'model_state_dict': model.state_dict(),
            	'optimizer_state_dict': optimizer.state_dict(),
            	'validation_accuray': acc.item()/len(val_loader),
            }, os.path.join(args.checkpoint_dir, 'model_epoch_'+str(epoch+1)+'.pt'))


	if args.tensorboard_display : 
		writer.flush()
		writer.close()
	logger.info('==> Finished Training')

	logger.info('==> Saving Checkpoint at Last Epoch')
	torch.save({
    	'epoch': epoch,
    	'iterations':iterations,
    	'model_state_dict': model.state_dict(),
    	'optimizer_state_dict': optimizer.state_dict(),
    	'validation_accuracy': acc.item()/len(val_loader),
    }, os.path.join(args.checkpoint_dir, 'final_model.pt'))



if __name__ == '__main__':
	main()