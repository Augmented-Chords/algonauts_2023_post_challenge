import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from scipy.stats import pearsonr as corr
from mlp import *
import torch.nn as nn
import torch.optim as optim

class argObj:
    def __init__(self, features_dir, parent_submission_dir, subj):

        self.subj = format(subj, '02')
        self.features_dir = features_dir
        self.subject_features_dir = os.path.join(self.features_dir,
            'subj'+self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
            'subj'+self.subj)

        if not os.path.isdir(self.subject_features_dir):
            os.makedirs(self.subject_features_dir)

# (train_loader, validation_loader, etc.)
def train_model(model,train_loader, validation_loader, device):
    model.to(device)
    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 50
    # Define loss function
    criterion = nn.MSELoss()

    # Define optimizer (e.g., Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
    
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        # Validation (optional)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            for val_inputs, val_targets in validation_loader:
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
 
            avg_val_loss = val_loss / len(validation_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Learning Rate: {learning_rate:.4f}")
            scheduler.step(val_loss)

def to_tensor(data_array, device):
    reshaped_data = np.reshape(data_array, (data_array.shape[0],1,data_array.shape[1]))
    tensor = torch.Tensor(reshaped_data).to(device) 
    return tensor

def to_array(tensor):
    try:
        cpu_tensor = tensor.to(torch.device("cpu")) 
        data_array = cpu_tensor.detach().numpy()
    except:
        data_array = tensor.detach().numpy()
    finally:
        reshaped_data = np.reshape(data_array, (data_array.shape[0],data_array.shape[2]))
        return reshaped_data

def main():
    parser = argparse.ArgumentParser(description="Use Multilayer Perceptron for fMRI data prediction")

    parser.add_argument('-s','--subject',type=int,default=8,help="select one subject (default: 8)")
    parser.add_argument('-d','--device',type=str,default='cuda',help="torch device (default: cuda)")
    parser.add_argument('-f','--features_path',type=str,default='algonauts_2023_features_concatenated',help="features path (default: algonauts_2023_features_concatenated)")
    parser.add_argument('-o','--output_path',type=str,default='algonauts_2023_challenge_submission',help="fmri prediction output path (default: algonauts_2023_challenge_submission)")

    parse_args = parser.parse_args()

    subj = parse_args.subject
    device = parse_args.device
    device = torch.device(device)
    features_dir = parse_args.features_path
    parent_submission_dir = parse_args.output_path

    args = argObj(features_dir, parent_submission_dir, subj)

    features_train = np.load(os.path.join(args.subject_features_dir, 'features_train.npy'))
    features_test = np.load(os.path.join(args.subject_features_dir, 'features_test.npy'))

    lh_fmri_train = np.load(os.path.join(args.subject_features_dir, 'lh_fmri_train.npy'))
    rh_fmri_train = np.load(os.path.join(args.subject_features_dir, 'rh_fmri_train.npy'))

    tensor_features_train = to_tensor(features_train, device)
    tensor_features_test = to_tensor(features_test, device)
    tensor_lh_train = to_tensor(lh_fmri_train, device)
    tensor_rh_train = to_tensor(rh_fmri_train, device)
    lh_train_dataset = TensorDataset(tensor_features_train,tensor_lh_train) # create datset
    rh_train_dataset = TensorDataset(tensor_features_train,tensor_rh_train)
    lh_train_dataloader = DataLoader(lh_train_dataset) # create dataloader
    rh_train_dataloader = DataLoader(rh_train_dataset)
    lh_val_dataloader = DataLoader(lh_train_dataset)
    rh_val_dataloader = DataLoader(rh_train_dataset)

    input_feature = features_train.shape[1]
    reg_lh_output_features = lh_fmri_train.shape[1]
    reg_rh_output_features = rh_fmri_train.shape[1]

    reg_lh = mlp(input_feature, input_feature * 2, reg_lh_output_features)
    print(reg_lh)
    train_model(reg_lh,lh_train_dataloader,lh_val_dataloader, device)

    reg_rh = mlp(input_feature, input_feature * 2, reg_rh_output_features)
    print(reg_rh)
    train_model(reg_rh,rh_train_dataloader,rh_val_dataloader, device)

    reg_lh.eval()  # Set the model to evaluation mode
    reg_rh.eval()
    lh_fmri_test_pred = to_array(reg_lh(tensor_features_test))
    rh_fmri_test_pred = to_array(reg_rh(tensor_features_test))

    lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
    rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
    np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

if __name__ == "__main__":
    main()