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
from sklearn.externals import joblib
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
        self.Training_Cars      = []
        self.Training_NonCars   = []
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
                    self.Training_Cars.append(os.path.join(path, name))

        for path, subdirs, files in os.walk(non_car_path):
            for name in files:
                filename, extension = os.path.splitext(name)
                if extension== '.png':
                    TrainIMG = Training_Image(Path = os.path.join(path, name), 
                                              image_class = self.image_classes.non_car)
                    self.Training_NonCars.append(os.path.join(path, name))

        print ("Training Images: Loaded %u %s and %u %s " %(len(self.Training_Cars), 
                                                            self.image_classes.car, 
                                                            len(self.Training_NonCars),
                                                            self.image_classes.non_car))

    def visualise_dataset(self):

        fig, axes = plt.subplots(4,4, figsize=(8, 8))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axes = axes.ravel()

        for i in range(8):
            image_path = self.Training_Cars[np.random.randint(0,len(self.Training_Cars))]
            img = mpimg.imread(image_path)
            axes[i].axis('off')
            axes[i].set_title(self.image_classes.car, fontsize=10)
            axes[i].imshow(img)
        for i in range(8,16):
            image_path  = self.Training_NonCars[np.random.randint(0,len(self.Training_NonCars))]
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
        
        t = time.time()
        # Use grid search to find best parameters  
        parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma': [0.1, 1, 10]}
        svr = svm.SVC()
        self.classifier = GridSearchCV(svr, parameters)
        t=time.time()
        self.classifier.fit(self.X_train, self.y_train)
        # training done
        self.training_time = time.time() - t
        print(' %2.2f Seconds to train SVC...' %(self.training_time))
        # Check the score of the SVC
        self.training_score = self.classifier.score(self.X_test, self.y_test)
        print('Test Accuracy of SVC = %2.4f' %(self.training_score))   

    def fit_classifier(self, Features = None, load = False):

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
        self.X_train    = X_train #X_scaler.transform(X_train)
        self.X_test     = X_test #X_scaler.transform(X_test)
        self.y_train    = y_train
        self.y_test     = y_test

        print('Feature vector length:', len(X_train[0]))

        if load:
            self.classifier = joblib.load('model.pkl')
        else:            
            t = time.time()
        
            # parameters = {'C':[0.1, 1, 10], 'gamma': [0.1, 1, 10]}
            # self.classifier = svm.SVC(verbose = True, kernel = 'linear', C = 1, gamma = 0.1)
            # svr = svm.SVC(verbose = True)
            self.classifier = svm.LinearSVC(verbose = True)
            # svr =svm.SVC(verbose = True)
            # self.classifier = GridSearchCV(svr, parameters)
            t=time.time()
            self.classifier.fit(self.X_train, self.y_train)
            # training done
            self.training_time = time.time() - t
            print(' %2.2f Seconds to train SVC...' %(self.training_time))
            joblib.dump(self.classifier, 'model.pkl') 

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

    def plot_summary(self, score_file = "training_scores.csv"):
        f =  open(score_file, 'r')
        reader = csv.reader(f, delimiter=',',)
        next(reader) # skip header
        training_scores = np.array([]).astype(float)
        training_time = np.array([]).astype(float)
        for row in reader:
            training_scores = np.append(training_scores, float(row[-1]))
            training_time = np.append(training_time, float(row[-2]))
        max_score = np.amax(training_scores)
        index = np.where(training_scores >= max_score)
        print (index)
        max_score = training_scores[index]
        max_score_time = training_time[index]    
        plt.plot(training_scores, 'b-o')
        plt.plot(index, max_score, 'r*')
        plt.grid()
        plt.ylabel("Training Score")
        plt.xlabel("Hyper Parameter Configuration")
        plt.show()

        plt.plot(training_time, 'b-o')
        plt.plot(index, max_score_time, 'r*')
        plt.grid()
        plt.ylabel("Training Time [sec]")
        plt.xlabel("Hyper Parameter Configuration")
        plt.show()


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
                       hist_bins = 16,
                       ystart = 400,
                       ystop =  656,
                       scale = 1.25):

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
        self.ystart = ystart 
        self.ystop =  ystop
        self.scale = scale
        self.classifier = None  
 
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
        
    def extract_features(self, imgs, image_class, sample_size = 500):
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

    def extract_img_features(self, image):
        image_features = []
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
            image_features.append(spatial_features)
        if self.hist_feat == True:
            # Apply color_hist()
            hist_features = self.color_hist(feature_image, nbins=self.hist_bins)
            image_features.append(hist_features)    

        # Call get_hog_features() with self.vis=False, self.feature_vec=True
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(self.get_hog_features(feature_image[:,:,channel]))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = self.get_hog_features(feature_image[:,:,self.hog_channel])
        # Append the new feature vector to the features list
        image_features.append(hog_features)
        return np.concatenate(image_features)
    

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

    def feature_visualiser(self, imgs):
        fig, axes = plt.subplots(2,4, figsize=(8, 8))
        fig.subplots_adjust(hspace = .2, wspace=.001)
        axes = axes.ravel()

        for i in range(0, 8, 2):
            image_path = imgs[np.random.randint(0,len(imgs))]
            img = mpimg.imread(image_path)
            gry = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            f, hog_img = hog(gry, orientations=self.orient,
                            pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                            cells_per_block=(self.cell_per_block, self.cell_per_block), visualise=True)
            axes[i].axis('off')
            axes[i].set_title("Image", fontsize=10, )
            axes[i].imshow(img, cmap='gray')
            axes[i+1].axis('off')
            axes[i+1].set_title("HOG Features", fontsize=10)
            axes[i+1].imshow(hog_img, cmap='gray')
            
        plt.show()

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def object_detection(self, img, draw = False):
        
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        
        ctrans_tosearch = img[self.ystart:self.ystop,:,:]
        if self.scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/self.scale), np.int(imshape[0]/self.scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        bounding_boxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step                
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                features = self.extract_img_features(subimg)
                # Scale features and make a prediction
                X = np.vstack(features)  
                test_prediction = TD.classifier.predict(X.reshape(1,-1))
                if test_prediction == 1:
                    xbox_left = np.int(xleft*self.scale)
                    ytop_draw = np.int(ytop*self.scale)
                    win_draw = np.int(window*self.scale)
                    bounding_box = ((xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart))
                    if draw:
                        cv2.rectangle(draw_img,(xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart),(0,0,255),2)
                    else:
                        bounding_boxes.append(bounding_box) 
        if draw:
            return draw_img
        else:
            return bounding_boxes

    def search_boxes(self, img):
        
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255
        
        ctrans_tosearch = img[self.ystart:self.ystop,:,:]
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        bounding_boxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step                
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                xbox_left = np.int(xleft*self.scale)
                ytop_draw = np.int(ytop*self.scale)
                win_draw = np.int(window*self.scale)
                bounding_box = ((xbox_left, ytop_draw+self.ystart),(xbox_left+win_draw,ytop_draw+win_draw+self.ystart))
                bounding_boxes.append(bounding_box)
        return bounding_boxes

    def process_frame(self, img):

        # far small
        self.ystart = 400
        self.ystop = 464
        self.scale = 0.8
        bounding_boxes = []
        bounding_boxes.append(self.object_detection(img))

        # far small
        self.ystart = 400
        self.ystop = 464
        self.scale = 1.0
        bounding_boxes = []
        bounding_boxes.append(self.object_detection(img))
        
        self.ystart = 416
        self.ystop = 480
        self.scale = 1.0
        bounding_boxes.append(self.object_detection(img))
        
        # far small
        self.ystart = 400
        self.ystop = 496
        self.scale = 1.5
        bounding_boxes.append(self.object_detection(img))
        
        self.ystart = 432
        self.ystop = 528
        self.scale = 1.5
        bounding_boxes.append(self.object_detection(img))
        
        # medium range
        self.ystart = 400
        self.ystop = 528
        self.scale = 2.0
        bounding_boxes.append(self.object_detection(img))
        
        self.ystart = 432
        self.ystop = 560
        self.scale = 2.0
        bounding_boxes.append(self.object_detection(img))
        
        # close range
        self.ystart = 400
        self.ystop = 596
        self.scale = 3.5
        bounding_boxes.append(self.object_detection(img))
       
        self.ystart = 464
        self.ystop = 660
        self.scale = 3.5
        bounding_boxes.append(self.object_detection(img))

        if len(bounding_boxes) > 0:
            bounding_boxes = [box for boxlist in bounding_boxes for box in boxlist] 
            heatmap = np.zeros_like(img[:,:,0])
            heatmap = self.add_heat(heatmap, bounding_boxes)
            heatmap = self.apply_threshold(heatmap, 1)
            labels = label(heatmap)
            draw_img = self.draw_labeled_bboxes(img, labels)
            return draw_img
        else:
            return img
    

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def draw_bboxes(self, img, boxes):
        drawimg = np.copy(img)
        # Iterate through all detected cars
        for bbox in boxes:
            # Draw the box on the image
            cv2.rectangle(drawimg, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return drawimg


Training_Mode = False # loop through all possible parameters
SHOW_IMAGE = False # show images for debugging and write up
LOAD_MODEL = True # load trained model 

if Training_Mode:
    """Parameters to test """

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

    result_file = "training_score_%s.csv" %(datetime.now())
    # Load Training Data
    TD = Training(result_file)
    TD.load_images()
    if SHOW_IMAGE:
        TD.visualise_dataset()

    for parameter_set in hyper_parameters: 
        t=time.time()
        print ("Parameters: %s, at %s" %(parameter_set, datetime.now())) 

        # extract test parameters
        (colorspace, orient, 
         pix_per_cell, cell_per_block, 
         hog_channel, spatial_feat, hist_feat, 
         spatial_size, hist_bin) = parameter_set

        Object_Detection = Object_Detection(
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

        Object_Detection.extract_features(TD.Training_Cars, TD.image_classes.car)
        Object_Detection.extract_features(TD.Training_NonCars, TD.image_classes.non_car)

        Object_Detection.extract_time = time.time() - t

        print(round(Object_Detection.extract_time, 2), 'Seconds to extract features...')

        # train a classifier with extracted features
        # use grid search to find optimal parameters
        TD.train_classifier(Features = Object_Detection)
        TD.output_score(Object_Detection)
    # clear memory 
    TD = None
    Object_Detection = None

else:
    """Final Parameters"""
    colorspace = "YUV"
    orient = 12
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL"
    spatial_feat = False
    hist_feat = False
    spatial_size = (16,16)
    hist_bin = 16
    result_file = None

    TD = Training(result_file)
    TD.load_images()
    if SHOW_IMAGE:
        TD.visualise_dataset()
    t=time.time()
    Object_Detection = Features(
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
    Object_Detection.extract_features(TD.Training_Cars, TD.image_classes.car, sample_size = 8500)
    Object_Detection.extract_features(TD.Training_NonCars, TD.image_classes.non_car, sample_size = 8500)

    Object_Detection.extract_time = time.time() - t

    print(round(Object_Detection.extract_time, 2), 'Seconds to extract features...')
        
    # Visualize HOG on example images
    if SHOW_IMAGE:
        Object_Detection.feature_visualiser(TD.Training_Cars)
    # train a classifier with extracted features
    # not need to use grid search
    if SHOW_IMAGE:
        TD.visualise_dataset()
    TD.fit_classifier(Features = Object_Detection, load = LOAD_MODEL) 
    Object_Detection.classifier = TD.classifier
    
    # use test image    
    if SHOW_IMAGE:
        files = os.listdir('test_images')
        for file in files:
            if file[-4:] == '.jpg':
                img = mpimg.imread('test_images/'+file)
                # draw = Object_Detection.object_detection(img, draw=True)
                draw = Object_Detection.process_frame(img)
                plt.imsave("boxes"+file[-5]+".png", draw)
    else:
        write = 'project_video_out.mp4'
        input_file = VideoFileClip('project_video.mp4')
        output_file = input_file.fl_image(Object_Detection.process_frame)
        output_file.write_videofile(write, audio=False)





