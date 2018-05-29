# libraries 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

# constants
# enables plotting and 
DEBUG = True
CALIBRATION_FOLDER = 'camera_cal'
TEST_FOLDER = 'test_images'


# lane finding class for correcting and processing images
class lane_finding():
    def __init__(self, show_images = False):
        # display images 
        self.show_images = show_images
        # image size
        self.im_size = None
        # camera calibration parameters
        self.dist = None
        self.mtx = None
        self.rvecs = None # rotation matrix
        self.tvecs = None # translation matrix

        # Gradient thresholds
        
        # Color thresholds
        
        # image perspective transform
        self.Minv = None
        
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

        # Perspective transform

        # Warp an image using the perspective transform, M:
        warped = cv2.warpPerspective(img, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

    # find lane 
    # params: processed image
    def lane_search(self, image):
        pass

    def transform_perspective(self, image_path):

        image = cv2.imread(image_path)
        
        src = np.float32([[1040, 680],[255, 680],[628, 431],[650, 431]])
        dst = np.float32([[1040, 680],[255, 680],[255, 430],[1040, 430]])
        # # Compute the inverse perspective transform:
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        height, width, channel = image.shape
        self.img_size = (height,width)
        top_down = cv2.warpPerspective(image, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

        if self.show_images:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            # ax1.plt(src,'.')
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(top_down)
            # ax2.plt(src,'.')
            ax2.set_title('Undistorted and Warped Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()



    # Camera calibration
    # params: calibration images folder path 
    def calibrate(self, calibration_folder):

        objpoints = [] # 3d space points
        imagepoints = [] # 2d points in image plane
        # list of files in directory
        image_names = os.listdir(calibration_folder)
        # board inner corner size
        nx,ny = 9,6

        for image_name in image_names:
            if image_name[-4:] == '.jpg': # process images only
                image_path = calibration_folder + '/' + image_name
                image = cv2.imread(image_path)
                # prepare points
                objp = np.zeros((nx*ny,3), np.float32)
                objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y 

                # find corners in 
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #Finding chessboard corners (for an 8x6 board):
                ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

                if ret:
                    imagepoints.append(corners)
                    objpoints.append(objp)

        #Camera calibration, given object points, image points, and the shape of the grayscale image:
        r, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, gray.shape[::-1], None, None)

        if self.show_images:
            #Drawing detected corners on an image:
            image = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            plt.imshow(image)
            plt.show()

            # Plotting thresholded images
            image_original = cv2.imread(calibration_folder + '/' + image_names[10])
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            f.tight_layout()
            ax1.set_title('Original Image')
            ax1.imshow(image_original)

            gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

            #Finding chessboard corners (for an 8x6 board):
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            dst = cv2.undistort(image_original, self.mtx, self.dist, None, self.mtx)

            ax2.set_title('Undistorted Image')
            ax2.imshow(dst)

            plt.show()

LF = lane_finding(DEBUG)

# LF.calibrate(CALIBRATION_FOLDER)
LF.transform_perspective(TEST_FOLDER+'/'+'straight_lines1.jpg')

if DEBUG:
    # LF.process_image()
    print ('Process here..')
else:
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    clip = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)



