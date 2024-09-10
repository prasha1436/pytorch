import torch
import torch.nn as nn

# model
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features= 6)

##1. Saving and loading a model
#FILE = "model.pth"

# saving the created (/trained, if trained) model
#torch.save(model.state_dict(), FILE)

# loading the saved model
#l_model = Model(n_input_features=6)
#l_model.load_state_dict(torch.load(FILE))
#l_model.eval()

#check results
#for params in l_model.parameters():
#    print(params)

##2. Saving and loading a checkpoint of a trained model

lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
print(optimizer.state_dict())

#creating and saving a checkpoint
checkpoint = {
    "epoch": 90,
    "model_state":model.state_dict(),
    "optim_state":optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")

# loading the checkpoint
l_checkpoint = torch.load("checkpoint.pth")
model_2 = Model(n_input_features=6)
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr = 0)

epoch = l_checkpoint["epoch"]
model_2.load_state_dict(l_checkpoint["model_state"])
optimizer_2.load_state_dict(l_checkpoint["optim_state"])
#check results
print("optimizer_2 = ", optimizer_2.state_dict())