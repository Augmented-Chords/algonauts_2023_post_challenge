import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr

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

def main():
    parser = argparse.ArgumentParser(description="Use Linear Regression for fMRI data prediction")

    parser.add_argument('-s','--subject',type=int,default=8,help="select one subject (default: 8)")
    parser.add_argument('-f','--features_path',type=str,default='algonauts_2023_features_concatenated',help="features path (default: algonauts_2023_features_concatenated)")
    parser.add_argument('-o','--output_path',type=str,default='algonauts_2023_challenge_submission',help="fmri prediction output path (default: algonauts_2023_challenge_submission)")

    parse_args = parser.parse_args()

    subj = parse_args.subject
    features_dir = parse_args.features_path
    parent_submission_dir = parse_args.output_path

    args = argObj(features_dir, parent_submission_dir, subj)

    features_train = np.load(os.path.join(args.subject_features_dir, 'features_train.npy'))
    features_test = np.load(os.path.join(args.subject_features_dir, 'features_test.npy'))
    lh_fmri_train = np.load(os.path.join(args.subject_features_dir, 'lh_fmri_train.npy'))
    rh_fmri_train = np.load(os.path.join(args.subject_features_dir, 'rh_fmri_train.npy'))

    # Fit linear regressions on the training data
    reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
    reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
    # Use fitted linear regressions to predict the validation and test fMRI data
    lh_fmri_test_pred = reg_lh.predict(features_test)
    rh_fmri_test_pred = reg_rh.predict(features_test)

    lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
    rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
    np.save(os.path.join(args.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(args.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

if __name__ == "__main__":
    main()