from torch.utils.data import Dataset
import numpy as np
import torch
""" 
Create a dataset for the bets. The dataset should have the following columns:
- Ace: Number of Aces in the deck
- Two: Number of Twos in the deck
- Three: Number of Threes in the deck
- Four: Number of Fours in the deck
- Five: Number of Fives in the deck
- Six: Number of Sixes in the deck
- Seven: Number of Sevens in the deck
- Eight: Number of Eights in the deck
- Nine: Number of Nines in the deck
- Ten: Number of Tens, Jacks, Queens, and Kings in the deck
- Reward: The reward for the bet
- Reward_class: Zero if the reward is negative or zero, one if the reward is positive
"""
class BetsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset.iloc[:,0:10].values
        self.y = dataset.iloc[:,10].values #use Reward as output
        self.rewards = dataset.iloc[:,10].values
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
  
def val_epoch(model, dataloader, loss, device):
  val_losses = []

  for idx,batch in enumerate(dataloader): # for each validation batch

    x, y = batch[0].to(device), batch[1].to(device).unsqueeze(1).float()

    preds = model(x)

    val_loss = loss(preds, y)
    val_losses.append(val_loss.item())

  return val_losses  
  
def train_and_validate(model, criterion, optimizer, dataloader_train, dataloader_test, epochs=100, loss_interval=25, device="cpu"):
    """
    Train and validate and model

    Arguments:
        model: model to train and validate
        criterion: loss function for the model
        optimizer: optimizer for the model
        dataloader_train: dataloader for train data
        dataloader_test: dataloader for test data
        epochs: number of epochs to train
        loss_interval: number of epochs between saving the training and validation loss.
        device: device on which to run the network
    

    Returns:
        (trained_model, train_loss, val_loss)
    """

    train_losses = []
    val_losses = []

    for i in range(epochs): # for each epoch
        t_losses = []
        for idx,batch in enumerate(dataloader_train): # for each batch

            x, y = batch[0].to(device), batch[1].to(device).unsqueeze(1).float()

            # clear gradients
            model.zero_grad()

            # predict
            #y_pred, hidden = model(batch[0], hidden)
            y_pred = model(x)

            # get loss
            loss = criterion(y_pred, y)
            t_losses.append(loss.item())
            loss = loss.mean()

            # propagate and train
            loss.backward()
            optimizer.step()
            

        if i%(loss_interval*2) == 0:
            print(i,"th epoch : ", np.mean(t_losses))

        # Validation and losses
        if i%loss_interval == 0:
            vlosses = val_epoch(model, dataloader_test, criterion, device)
            val_losses.append(np.mean(vlosses))
            train_losses.append(np.mean(t_losses))
            print(f"Validation loss for epoch {i}: {np.mean(vlosses)}")

    return model, train_losses, val_losses