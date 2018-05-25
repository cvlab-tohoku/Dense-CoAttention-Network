
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch.nn as nn

from torchvision import models


class ResNet(nn.Module):

	def __init__(self, name, is_freeze=True):
		assert name in ["resnet50", "resnet101", "resnet152"], "Invalid CNN type!"
		super(ResNet, self).__init__()

		self.model = models.__dict__[name](pretrained=True)
		delattr(self.model, "fc")
		delattr(self.model, "avgpool")

		if is_freeze:
			print("Freezing %s ..." % name)
			for param in self.model.parameters():
				param.requires_grad = False

	def forward(self, image):
		image = self.model.conv1(image)
		image = self.model.bn1(image)
		image = self.model.relu(image)
		image = self.model.maxpool(image)

		output1 = self.model.layer1(image)
		output2 = self.model.layer2(output1)
		output3 = self.model.layer3(output2)
		output4 = self.model.layer4(output3)

		return output1, output2, output3, output4