import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
import argparse
import os


class MNISTLoader:
	def __init__(self, batch_size):
		self.data_transforms = transforms.Compose([transforms.ToTensor()])
		self.batch_size = batch_size
		self.test, self.train = self.get_data()

	def get_data(self):
		trainset = datasets.MNIST(root='./data', train=True, download=True, transform=self.data_transforms)
		testset = datasets.MNIST(root='./data', train=False, download=True, transform=self.data_transforms)
		train = DataLoader(trainset, batch_size=self.batch_size,shuffle=True)
		test = DataLoader(testset, batch_size=self.batch_size,shuffle=False)
		return test, train



class Model(nn.Module):
	def __init__(self):
		super (Model, self).__init__()
		self.net = nn.Sequential(nn.Linear(784, 300), nn.ReLU(inplace=True),nn.Linear(300, 300),nn.ReLU(inplace=True),nn.Linear(300, 200),nn.ReLU(inplace=True),nn.Linear(200, 10),nn.Softmax(dim=1))
		self.criterion = nn.NLLLoss()	

	def forward(self, x):
		x = self.net(x)
		return x


	def train(self, lr, epochs, dataset):
		print("Training:")
		for epoch in range(epochs):
			criterion = self.criterion
			optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
			for images, labels in tqdm(dataset.train, desc="Epoch " + str(epoch+1)+'/'+str(epochs)):
				images = images.view(images.shape[0], -1)
				optimizer.zero_grad()
				output = self(images)
				loss = criterion(output, labels)
				loss.backward()
				optimizer.step()

	def evaluate(self, dataset):
		correct, total = 0, 0
		for imgs, labels in tqdm(dataset.test, desc="Evaluating"):
			for i in range(len(labels)):
				img = imgs[i].view(1, 784)
				with torch.no_grad():
					logps = self(img)
				ps = torch.exp(logps)
				prob = list(ps.numpy()[0])
				predict = prob.index(max(prob))
				true = labels.numpy()[i]
				if predict == true:
					correct += 1
				total += 1
		acc = correct/total
		print("Test accuracy:", acc)
		return acc

	def save(self, output):
		torch.save(model.state_dict(), output)
		
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="MNIST", type=str, required=False, help="Only MNIST option is available now")
	parser.add_argument("--model", default=None, type=str, required=False, help="If you want to evaluate already trained model, pass a path to the model here")
	parser.add_argument("--batch_size", default=64, type=int, required=False, help="Batch size for training and test")
	parser.add_argument("--output_dir", default="./output", type=str, required=False, help="Where to save the model")
	parser.add_argument("--epochs", default=15, type=int, required=False, help="Number of epochs")
	parser.add_argument("--train", action="store_true", help="If you need to train")
	parser.add_argument("--eval", action="store_true", help="If you need to evaluate")
	parser.add_argument("--learning_rate", default=0.003, type=float, help="Learning rate")
	parser.add_argument("--seed", default=2020, type=int, help="Set a seed")
	args = parser.parse_args()
	
	torch.manual_seed(args.seed)

	if args.dataset == "MNIST":
		data = MNISTLoader(batch_size=args.batch_size)
	else:
		print("The Data Loader for the dataset", args.dataset, "is not implemented")
		exit()

	if args.train:
		model = Model()
		model.train(args.learning_rate, args.epochs, data)
		try:
			os.mkdir(args.output_dir)
		except:
			pass
		model.save(args.output_dir+"/model.pt")
	
	if args.model is not None:
		model = Model()
		model.load_state_dict(torch.load(args.model), strict=False)

	if not args.train and not args.model:
		print("You haven't chosen or trained a model")
		exit()	

	if args.eval:
		model.evaluate(data)
