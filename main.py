import math
import torch
import torchcde

def train(model, num_epochs: int, data, optimizer, interpolation, loss_func):
  train_X, train_Y = data
  train_coeffs = interpolation(train_X)

  train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_Y)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

  for epoch in range(num_epochs):
      for batch in train_dataloader:
          batch_coeffs, batch_y = batch
          pred_y = model(batch_coeffs).squeeze(-1)
          loss = loss_func(pred_y, batch_y)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
      print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

def test(model, test_data, interpolation):
  test_X, test_Y = test_data
  pred = model(test_X).squeeze(-1)
  prediction_matches = (pred == test_Y).to(test_Y.dtype)
  proportion_correct = prediction_matches.sum() / test_Y.size(0)

  return proportion_correct

def main():
  #Hyperparameters
  num_epochs = 30

  #Get Data Here

  #Initialize Model
  model = NeuralCDE(input_channels=3, hidden_channels=8, output_channels=1)

  #Optimizer
  optimizer = torch.optim.Adam(model.parameters())

if __name__ == '__main__':
  main()
