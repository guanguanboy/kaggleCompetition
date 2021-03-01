import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions
from dataset import get_data
from torch.utils.data import DataLoader

class NN(nn.Module):
    def __init__(self, input_size):
        super(NN,self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NN(input_size=200).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn = nn.BCELoss()
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

x, y = next(iter(train_loader))
print(x.shape)

for epoch in range(1):
    propabilities, true = get_predictions(val_loader, model, device=DEVICE)
    print(f"VALIDATION ROC: {metrics.roc_auc_score(true, propabilities)}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        #forward
        scores = model(data)
        #print(scores.shape)

        loss = loss_fn(scores, targets)
        #print(loss)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


