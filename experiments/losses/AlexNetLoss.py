import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class AlexNetLoss(nn.Module):
    def __init__(self, device, loss_weight=1.0):
        super(AlexNetLoss, self).__init__()
        original_model = models.alexnet(pretrained=True).to(device)
        self.modified_model = ModifiedAlexNet(original_model).to(device)
        self.modified_model.eval()
        self.loss_fn = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, input1, input2):
        with torch.no_grad():
            if input1.size(1) == 1:  # Grayscale image
                input1 = input1.repeat(1, 3, 1, 1)  # Repeat channel 3 times

            if input2.size(1) == 1:  # Grayscale image
                input2 = input2.repeat(1, 3, 1, 1)  # Repeat channel 3 times

            output1, intermediate_outputs1 = self.modified_model(input1)
            output2, intermediate_outputs2 = self.modified_model(input2)

        loss = 0
        for output1, output2 in zip(intermediate_outputs1, intermediate_outputs2):
            loss += self.loss_fn(output1, output2)
        return loss * self.loss_weight


class ModifiedAlexNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedAlexNet, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier

    def forward(self, x):
        intermediate_outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [4, 9]:  # Collect outputs after layers 4 and 9
                intermediate_outputs.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x, intermediate_outputs