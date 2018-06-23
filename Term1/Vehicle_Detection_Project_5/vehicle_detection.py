#!/usr/bin/env python3

#Load libraries 
from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label
from sklearn.model_selection import GridSearchCV
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import cv2
import glob
import time
from datetime import datetime
import os
import csv
import collections
import itertools
from enum import Enum 


"""single image class"""
class Training_Image:
    def __init__(self, 
                 Path = " ", 
                 image_class = None):
        self.image_path = Path
        self.image_class = image_class

"""Training Dataset"""
class Training:
    def __init__(self, result_file = None):
        self.Training_Cars      = {}
        self.Training_NonCars   = {}
        image_classes           = collections.namedtuple('image_classes', 
                                       ['car', 'non_car'])
        self.image_classes      = image_classes('car', 'non_car')

        # training dataset features and labels
        self.X_train    = None
        self.X_test     = None
        self.y_train    = None
        self.y_test     = None
        self.classifier = None
        self.training_time = 0.0

        if result_file != None:   
            # init output file if needed
            self.result_file = result_file
            row = ["Features.cspace",
            "Features.orient",
            "Features.pix_per_cell",
            "Features.cell_per_block",
            "Features.hog_channel",
            "Features.extract_time",
            "Features.spatial_feat", 
            "Features.hist_feat",
            "Features.spatial_size",
            "Features.hist_bins",
            "classifier.best_params_",
            "training_time",
            "training_score"]

            File = open(self.result_file, 'w')  
            with File:  
               writer = csv.writer(File)
               writer.writerow(row)

    def load_images(self, car_path = "vehicles", non_car_path = "non-vehicles"):
        # get car images from folder car_path
        car_path     = os.path.join(os.getcwd(), car_path)
        non_car_path = os.path.join(os.getcwd(), non_car_path)

        for path, subdirs, files in os.walk(car_path):
            for name in files:
                filename, extension = os.path.splitext(name)
                if extension== '.png':
                    TrainIMG = Training_Image(Path = os.path.join(path, name), 
                                              image_class = self.image_classes.car)
                    self.Training_Cars.update({TrainIMG.image_path: TrainIMG})

        for path, subdirs, files in os.walk(non_car_path):
            for name in files:
                filename, extension = os.path.splitext(name)
                if extension== '.png':
                    TrainIMG = Training_Image(Path = os.path.join(path, name), 
                                              image_class = self.image_classes.non_car)
                    self.Training_NonCars.update({TrainIMG.image_path: TrainIMG})

        print ("Training Images: Loaded %u %s and %u %s " %(len(self.Training_Cars), 
                                                            self.image_classes.car, 
                                                            len(self.Training_NonCars),
                                                            self.image_classes.non_car))

    def visualise_dataset(self):

        fig, axes = plt.subplots(4,4, figsize=(8, 8))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axes = axes.ravel()

        for i in range(8):
            image_path = list(self.Training_Cars)[np.random.randint(0,len(self.Training_Cars))]
            img = mpimg.imread(image_path)
            axes[i].axis('off')
            axes[i].set_title(self.image_classes.car, fontsize=10)
            axes[i].imshow(img)
        for i in range(8,16):
            image_path  = list(self.Training_NonCars)[np.random.randint(0,len(self.Training_Cars))]
            img = mpimg.imread(image_path)
            axes[i].axis('off')
            axes[i].set_title(self.image_classes.non_car, fontsize=10)
            axes[i].imshow(img)
        plt.show()

    def train_classifier(self, Features = None):
        # Create an array stack of feature vectors
        X = np.vstack((Features.features_car, 
                      Features.features_non_car)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(Features.features_car)), np.zeros(len(Features.features_non_car))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)
            
        # Fit a per-column scaler # normalize
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        self.X_train    = X_scaler.transform(X_train)
        self.X_test     = X_scaler.transform(X_test)
        self.y_train    = y_train
        self.y_test     = y_test

        print('Feature vector length:', len(X_train[0]))
        
        # Use grid search to find best parameters  
        parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma': [0.1, 1, 10]}
        svr = svm.SVC()
        self.classifier = GridSearchCV(svr, parameters)
        t=time.time()
        self.classifier.fit(X_train, y_train)
        # training done
        self.training_time = time.time() - t
        print(' %2.2f Seconds to train SVC...' %(self.training_time))
        # Check the score of the SVC
        self.training_score = self.classifier.score(self.X_test, self.y_test)
        print('Test Accuracy of SVC = %2.4f' %(self.training_score))        

    def output_score(self, Features):
        if self.result_file != None:
            row = [Features.cspace,
                    Features.orient,
                    Features.pix_per_cell,
                    Features.cell_per_block,
                    Features.hog_channel,
                    Features.extract_time,
                    Features.spatial_feat, 
                    Features.hist_feat,
                    Features.spatial_size,
                    Features.hist_bins,
                    self.classifier.best_params_,
                    self.training_time,
                    self.training_score]

            File = open(self.result_file, 'a')  
            with File:  
               writer = csv.writer(File)
               writer.writerow(row)
        else:
            print ("Logging skipped")



"""Feature Extraction"""
class Features:
    def __init__(self, cspace='RGB',
                       orient=9, 
                       pix_per_cell=8, 
                       cell_per_block=2,
                       hog_channel=0, 
                       vis=False, 
                       feature_vec=True,
                       spatial_feat = False,
                       hist_feat = False,
                       spatial_size = (32,32),
                       hist_bins = 16):

        self.cspace=cspace
        self.orient=orient
        self.pix_per_cell=pix_per_cell 
        self.cell_per_block=cell_per_block
        self.hog_channel=hog_channel
        self.vis=vis
        self.feature_vec=feature_vec
        self.features_car = None
        self.features_non_car = None
        self.extract_time = None
        self.spatial_feat = spatial_feat
        self.hist_feat    = hist_feat
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
 
    # Define a function to compute binned color features  
    def bin_spatial(self, img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
        
    def extract_features(self, imgs_dict, image_class, sample_size = 500):
        # create an list to index
        imgs = list(imgs_dict)[0:sample_size]
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = mpimg.imread(file)
            # apply color conversion if other than 'RGB'
            if self.cspace != 'RGB':
                if self.cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif self.cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif self.cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif self.cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif self.cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)  

            if self.spatial_feat == True:
                spatial_features = self.bin_spatial(feature_image, size=self.spatial_size)
                file_features.append(spatial_features)
            if self.hist_feat == True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image, nbins=self.hist_bins)
                file_features.append(hist_features)    

            # Call get_hog_features() with self.vis=False, self.feature_vec=True
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:,:,channel]))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel])
            # Append the new feature vector to the features list
            file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        if image_class == Training().image_classes.car:
            self.features_car = features
        elif image_class == Training().image_classes.non_car:
            self.features_non_car = features

        print ("Extracted %u %s features" %(len(features), image_class))

    def get_hog_features(self, img):
        # Call with two outputs if self.vis==True
        if self.vis == True:
            features, hog_image = hog(img, 
                                      orientations=self.orient, 
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block), 
                                      transform_sqrt=False, 
                                      visualise=self.vis, feature_vector=self.feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(img, 
                          orientations=self.orient, 
                          pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                          cells_per_block=(self.cell_per_block, self.cell_per_block), 
                          transform_sqrt=False, 
                          visualise=self.vis, feature_vector=self.feature_vec)
            return features

"""Parameters """
Training_Mode = True
Show_Image = False

colorspaces = ["RGB", "HSV", "LUV"]
orients = [12]
pix_per_cells = [8, 16]
cell_per_blocks = [2]
hog_channels = [0,1,2,"ALL"]
spatial_feats = [True]
hist_feats = [True]
spatial_sizes = [(16,16), (32,32)]
hist_bins = [16, 32]

hyper_parameters = itertools.product(colorspaces, 
                                    orients,
                                    pix_per_cells,
                                    cell_per_blocks,
                                    hog_channels,
                                    spatial_feats,
                                    hist_feats,
                                    spatial_sizes,
                                    hist_bins)

if Training_Mode:
    result_file = "training_score_%s.csv" %(datetime.now())
else:
    result_file = None

if Training_Mode:
    # Load Training Data
    TD = Training(result_file)
    TD.load_images()
    if Show_Image:
        TD.visualise_dataset()

    for parameter_set in hyper_parameters: 
        t=time.time()
        print ("Parameters: %s, at %s" %(parameter_set, datetime.now())) 

        # extract test parameters
        (colorspace, orient, 
         pix_per_cell, cell_per_block, 
         hog_channel, spatial_feat, hist_feat, 
         spatial_size, hist_bin) = parameter_set

        Training_Features = Features(
                                    cspace=colorspace, 
                                    orient=orient, 
                                    pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat,
                                    spatial_size = spatial_size,
                                    hist_bins = hist_bin
                                    )

        Training_Features.extract_features(TD.Training_Cars, TD.image_classes.car)
        Training_Features.extract_features(TD.Training_NonCars, TD.image_classes.non_car)

        Training_Features.extract_time = time.time() - t

        print(round(Training_Features.extract_time, 2), 'Seconds to extract features...')

        # train a classifier with extracted features
        TD.train_classifier(Features = Training_Features)
        TD.output_score(Training_Features)
    # clear memory 
    TD = None
    Training_Features = None


else:
    pass
# Visualize HOG on example image




