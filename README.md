# algonauts_2023_post_challenge

## extract_features.py

```
usage: extract_features.py [-h] [-s SUBJECT] [-d DEVICE] [-r RAND_SEED] [-b BATCH_SIZE] [-n N_COMPONENTS]
                           [-v VALIDATION_RATIO] [-p DATA_PATH] [-o OUTPUT_PATH]

Extracts features from all ConvNeXt Blocks in the pretrained ConvNeXt-T and performs Incremental PCA

options:
  -h, --help            show this help message and exit
  -s SUBJECT, --subject SUBJECT
                        select one subject (default: 8)
  -d DEVICE, --device DEVICE
                        torch device (default: cuda)
  -r RAND_SEED, --rand_seed RAND_SEED
                        random seed (default: 5)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size (default: 128)
  -n N_COMPONENTS, --n_components N_COMPONENTS
                        pca n_components, must be less or equal to the batch number of samples (default: 128)
  -v VALIDATION_RATIO, --validation_ratio VALIDATION_RATIO
                        ratio for validation (default: 0.1)
  -p DATA_PATH, --data_path DATA_PATH
                        path of algonauts 2023 challenge data (default: algonauts_2023_challenge_data)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        output path of features (default: algonauts_2023_features_concatenated)
```

## linear_regression.py

```
usage: linear_regression.py [-h] [-s SUBJECT] [-f FEATURES_PATH] [-o OUTPUT_PATH]

Use Linear Regression for fMRI data prediction

options:
  -h, --help            show this help message and exit
  -s SUBJECT, --subject SUBJECT
                        select one subject (default: 8)
  -f FEATURES_PATH, --features_path FEATURES_PATH
                        features path (default: algonauts_2023_features_concatenated)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        fmri prediction output path (default: algonauts_2023_challenge_submission)
```

## multilayer_perceptron_submission.py

```
usage: multilayer_perceptron_submission.py [-h] [-s SUBJECT] [-d DEVICE] [-f FEATURES_PATH] [-o OUTPUT_PATH]

Use Multilayer Perceptron for fMRI data prediction

options:
  -h, --help            show this help message and exit
  -s SUBJECT, --subject SUBJECT
                        select one subject (default: 8)
  -d DEVICE, --device DEVICE
                        torch device (default: cuda)
  -f FEATURES_PATH, --features_path FEATURES_PATH
                        features path (default: algonauts_2023_features_concatenated)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        fmri prediction output path (default: algonauts_2023_challenge_submission)
```
