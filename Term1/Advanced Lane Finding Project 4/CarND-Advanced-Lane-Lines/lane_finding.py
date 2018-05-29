import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

DEBUG = True

class lane_finding():
    def __init__(self, show_images = False):
        # display images 
        self.show_images = show_images
        # camera calibration parameters
        self.dist = None
        self.mtx = None
        self.rvecs = None # rotation matrix
        self.tvecs = None # translation matrix

        # Gradient thresholds
        
        # Color thresholds
        

    def process_image(self, img):

        # find corners in 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        #Correcting the test image:
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


        #Distortion correction

        #Color/gradient threshold
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # Grayscale image
        # NOTE: we already saw that standard grayscaling lost color information for the lane lines
        # Explore gradients in other colors spaces / color channels to see what might work better

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')

        #Perspective transform
        #Compute the inverse perspective transform:
        Minv = cv2.getPerspectiveTransform(dst, src)
        #Warp an image using the perspective transform, M:
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    #Camera calibration
    def calibrate(self, calibration_folder):

        objpoints = [] # 3d space points
        imagepoints = [] # 2d points in image plane

        image_names = os.listdir(calibration_folder)

        nx,ny = 9,6

        for image_name in image_names:
            if image_name[-4:] == '.jpg':
                image_path = calibration_folder + '/' + image_name
                image = cv2.imread(image_path)
                # prepare points
                objp = np.zeros((6*9,3), np.float32)
                objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y 

                # find corners in 
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #Finding chessboard corners (for an 8x6 board):
                ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

                #Drawing detected corners on an image:
                # image = cv2.drawChessboardCorners(image, (9,6), corners, ret)

                if ret:
                    imagepoints.append(corners)
                    objpoints.append(objp)

        #Camera calibration, given object points, image points, and the shape of the grayscale image:
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imagepoints, gray.shape[::-1], None, None)

        if self.show_images:
            # Plotting thresholded images
            image_original = cv2.imread(calibration_folder + '/' + image_names[10])
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            ax1.set_title('Original Image')
            ax1.imshow(image_original)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #Finding chessboard corners (for an 8x6 board):
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            dst = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

            ax2.set_title('Undistorted Image')
            ax2.imshow(dst)

            plt.show()

LF = lane_finding(DEBUG)

LF.calibrate('camera_cal')
# white_output = 'test_videos_output/solidWhiteRight.mp4'
# ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# ## To do so add .subclip(start_second,end_second) to the end of the line below
# ## Where start_second and end_second are integer values representing the start and end of the subclip
# ## You may also uncomment the following line for a subclip of the first 5 seconds
# ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)



