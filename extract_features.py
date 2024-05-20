import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from nilearn import datasets
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from torchvision import models
from sklearn.decomposition import IncrementalPCA

class argObj:
  def __init__(self, data_dir, features_dir, subj):

    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
    self.features_dir = features_dir
    self.subject_features_dir = os.path.join(self.features_dir,
        'subj'+self.subj)

    if not os.path.isdir(self.subject_features_dir):
        os.makedirs(self.subject_features_dir)

class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform, device):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(self.device)
        return img

def fit_pca(feature_extractor, dataloader, n_components, batch_size):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca

def extract_features(feature_extractor, dataloader, pca):
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)

def extract_by_layer(model, model_layer, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, n_components, batch_size):
    feature_extractor = create_feature_extractor(model, return_nodes=model_layer)
    print("Next: Fit PCA transformer")
    pca = fit_pca(feature_extractor, train_imgs_dataloader, n_components, batch_size)
    print("Next: Apply PCA transform")
    features_train = extract_features(feature_extractor, train_imgs_dataloader, pca)
    features_val = extract_features(feature_extractor, val_imgs_dataloader, pca)
    features_test = extract_features(feature_extractor, test_imgs_dataloader, pca)
    del model, pca, feature_extractor
    torch.cuda.empty_cache()
    return features_train, features_val, features_test

def main():
    parser = argparse.ArgumentParser(description="Extracts features from all ConvNeXt Blocks in the pretrained ConvNeXt-T and performs Incremental PCA")

    parser.add_argument('-s','--subject',type=int,default=8,help="select one subject (default: 8)")
    parser.add_argument('-d','--device',type=str,default='cuda',help="torch device (default: cuda)")
    parser.add_argument('-r','--rand_seed',type=int,default=5,help="random seed (default: 5)") 
    parser.add_argument('-b','--batch_size',type=int,default=128,help="batch size (default: 128)")
    parser.add_argument('-n','--n_components',type=int,default=128,help="pca n_components, must be less or equal to the batch number of samples (default: 128)")
    parser.add_argument('-v','--validation_ratio',type=float,default=0.1,help="ratio for validation (default: 0.1)")
    parser.add_argument('-p','--data_path',type=str,default='algonauts_2023_challenge_data',help="path of algonauts 2023 challenge data (default: algonauts_2023_challenge_data)")
    parser.add_argument('-o','--output_path',type=str,default='algonauts_2023_features_concatenated',help="output path of features (default: algonauts_2023_features_concatenated)")   

    parse_args = parser.parse_args()

    subj = parse_args.subject

    device = parse_args.device
    device = torch.device(device)

    rand_seed = parse_args.rand_seed
    batch_size = parse_args.batch_size
    np.random.seed(rand_seed)
    n_components = parse_args.n_components
    validation_ratio = parse_args.validation_ratio
    data_dir = parse_args.data_path
    features_dir = parse_args.output_path

    args = argObj(data_dir, features_dir, subj)

    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    print('Subject ' + str(subj))

    print('LH training fMRI data shape:')
    print(lh_fmri.shape)
    print('(Training stimulus images × LH vertices)')

    print('\nRH training fMRI data shape:')
    print(rh_fmri.shape)
    print('(Training stimulus images × RH vertices)')

    train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    print('Test images: ' + str(len(test_img_list)))

    # n_components must be less or equal to the batch number of samples
    # num_train = int(len(train_img_list) * (1.0 - validation_ratio)) - (int(len(train_img_list) * (1.0 - validation_ratio)) % n_components)
    num_train = int(len(train_img_list) * (1.0 - validation_ratio))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    print('\nTest stimulus images: ' + format(len(idxs_test)))

    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])

    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform, device),
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform, device),
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform, device),
        batch_size=batch_size
    )

    lh_fmri_train = lh_fmri[idxs_train]
    rh_fmri_train = rh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_val = rh_fmri[idxs_val]

    del lh_fmri, rh_fmri

    model = models.convnext_tiny(pretrained=True)
    model.to(device) # send the model to the chosen device ('cpu' or 'cuda')
    model.eval() # set the model to evaluation mode, since you are not training it
    # train_nodes, _ = get_graph_node_names(model)
    # print(train_nodes)

    model_layer_1 = {'features.1.0.add':'1.0','features.1.1.add':'1.1','features.1.2.add':'1.2'}
    model_layer_3 = {'features.3.0.add':'3.0','features.3.1.add':'3.1','features.3.2.add':'3.2'}
    model_layer_5 = {'features.5.0.add':'5.0','features.5.1.add':'5.1','features.5.2.add':'5.2','features.5.3.add':'5.3','features.5.4.add':'5.4','features.5.5.add':'5.5','features.5.6.add':'5.6','features.5.7.add':'5.7','features.5.8.add':'5.8'}
    model_layer_7 = {'features.7.0.add':'7.0','features.7.1.add':'7.1','features.7.2.add':'7.2'}
    
    print('Extracts features from all ConvNeXt Blocks in Stage 1:')
    features_train_1, features_val_1, features_test_1 = extract_by_layer(model, model_layer_1, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, n_components, batch_size)
    print('Extracts features from all ConvNeXt Blocks in Stage 2:')
    features_train_3, features_val_3, features_test_3 = extract_by_layer(model, model_layer_3, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, n_components, batch_size)
    print('Extracts features from all ConvNeXt Blocks in Stage 3:')
    features_train_5, features_val_5, features_test_5 = extract_by_layer(model, model_layer_5, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, n_components, batch_size)
    print('Extracts features from all ConvNeXt Blocks in Stage 4:')
    features_train_7, features_val_7, features_test_7 = extract_by_layer(model, model_layer_7, train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader, n_components, batch_size)

    features_train = np.concatenate((features_train_1, features_train_3, features_train_5, features_train_7), axis=1)
    features_val = np.concatenate((features_val_1, features_val_3, features_val_5, features_val_7), axis=1)
    features_test = np.concatenate((features_test_1, features_test_3, features_test_5, features_test_7), axis=1)

    np.save(os.path.join(args.subject_features_dir, 'features_train.npy'), features_train)
    np.save(os.path.join(args.subject_features_dir, 'features_val.npy'), features_val)
    np.save(os.path.join(args.subject_features_dir, 'features_test.npy'), features_test)
    np.save(os.path.join(args.subject_features_dir, 'lh_fmri_train.npy'), lh_fmri_train)
    np.save(os.path.join(args.subject_features_dir, 'lh_fmri_val.npy'), lh_fmri_val)
    np.save(os.path.join(args.subject_features_dir, 'rh_fmri_train.npy'), rh_fmri_train)
    np.save(os.path.join(args.subject_features_dir, 'rh_fmri_val.npy'), rh_fmri_val)

    print('Extracted training features data shape:')
    print(np.load(os.path.join(args.subject_features_dir, 'features_train.npy')).shape)
    print('Extracted validation features data shape:')
    print(np.load(os.path.join(args.subject_features_dir, 'features_val.npy')).shape)
    print('Extracted test features data shape:')
    print(np.load(os.path.join(args.subject_features_dir, 'features_test.npy')).shape)
    print('All done')

if __name__ == "__main__":
    main()
