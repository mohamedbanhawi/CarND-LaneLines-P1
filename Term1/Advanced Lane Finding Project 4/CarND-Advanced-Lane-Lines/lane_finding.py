# libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import os
import time


# constants
# enables plotting and 
DEBUG = True
TRACKING = False
CALIBRATION_FOLDER = 'camera_cal'
TEST_FOLDER = 'test_images'


# lane finding class for correcting and processing images
class lane_finding():
    def __init__(self, show_images = False, tracking = False):
        # display images 
        self.show_images = False
        # tracking state
        self.enable_tracking = tracking
        self.tracking_left = False
        self.tracking_right = False
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension  
        
    def process_image(self, image):
        
        ################## Lens Distortion Correction ##################
        corrected_image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        image_unmodified = np.copy(corrected_image)

        ################## Gradient Thresholds ##################
        # Grayscale image
        gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)

        # Gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in y
        # abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        # scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Gradient direction
        min_angle = 40*np.pi/180
        max_angle = 75*np.pi/180
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        graddir_binary =  np.zeros_like(absgraddir)
        graddir_binary[(absgraddir >= min_angle) & (absgraddir <= max_angle)] = 1
        # gradient magnitude
        mag_thresh=(50, 180)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        gradmag_binary = np.zeros_like(gradmag)
        gradmag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        ################## Color threshold ##################
        # R-channel
        r_channel = corrected_image[:,:,2]
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(corrected_image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # Threshold R- color channel
        r_thresh_min = 200
        r_thresh_max = 255
        r_binary = np.zeros_like(r_channel)
        r_binary[(r_channel >= r_thresh_min) & (r_channel <= r_thresh_max)] = 1

        # Threshold S- color channel
        s_thresh_min = 125
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Combine the binary thresholds
        combined_binary = np.zeros_like(graddir_binary)
        combined_binary[((graddir_binary == 1) & (gradmag_binary ==1)) | ((s_binary == 1) & (r_binary == 1))] = 1

        ################## Perspective Warp ##################
        # Warp an image using the perspective transform, M:
        warped = cv2.warpPerspective(combined_binary, self.M, self.img_size, flags=cv2.INTER_LINEAR)

        if self.show_images:
            # Plotting thresholded images
            f, axes = plt.subplots(2, 3, figsize=(20,10))
            axes[0,0].set_title('S-Channel')
            axes[0,0].imshow(s_binary, cmap='gray')
            axes[0,1].set_title('R- channel')
            axes[0,1].imshow(r_binary, cmap='gray')
            axes[0,2].set_title('Gradient Direction')
            axes[0,2].imshow(graddir_binary, cmap='gray')
            axes[1,0].set_title('Gradient magnitude')
            axes[1,0].imshow(gradmag_binary, cmap='gray')
            axes[1,1].set_title('Combined ')
            axes[1,1].imshow(combined_binary, cmap='gray')
            axes[1,2].set_title('Original')
            axes[1,2].imshow(corrected_image)
            plt.show()

        # sliding window lane fit
        out_img = self.lane_search(warped)

        rgb_warp = cv2.warpPerspective(image_unmodified, self.M, self.img_size, flags=cv2.INTER_LINEAR)

        # Recast the x and y points into usable format for cv2.fillPoly()
        if self.tracking_left and self.tracking_right:
            right = np.array([np.transpose(np.vstack([self.right_fitx, self.fity]))])
            left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, self.fity])))])
            pts = np.hstack((left, right))
            cv2.fillPoly(rgb_warp, np.int_([pts]), (0,0,255))
        if self.tracking_left: 
            left = np.array([np.transpose(np.vstack([self.left_fitx, self.fity]))])
            cv2.polylines(rgb_warp, np.int_([left]),0, (255,0,0),20)
        if self.tracking_left:
            right = np.array([np.transpose(np.vstack([self.right_fitx, self.fity]))])
            cv2.polylines(rgb_warp, np.int_([right]),0, (255,0,0),20)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        image_info = cv2.warpPerspective(rgb_warp, self.Minv, self.img_size, flags=cv2.INTER_LINEAR)

        small_img = cv2.resize(out_img, (self.img_size[0]//4, self.img_size[1]//4))

        small_binary = cv2.resize(combined_binary, (self.img_size[0]//4, self.img_size[1]//4))
        small_gray = np.zeros_like(small_img)
        small_gray[:,:,0] = small_binary*255
        small_gray[:,:,1] = small_binary*255
        small_gray[:,:,2] = small_binary*255
        x_offset=50
        y_offset=50
        corrected_image[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img
        x_offset=self.img_size[0] - 50 - self.img_size[0]//4
        corrected_image[y_offset:y_offset+small_binary.shape[0], x_offset:x_offset+small_binary.shape[1]] = small_gray

        # Combine the result with the original image
        result = cv2.addWeighted(corrected_image, 1, image_info, 0.3, 0)
        cv2.putText(result, self.rad_text, (420,self.img_size[1]//8,),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)


        if True:#self.show_images:
            result = result[...,::-1]
            plt.imshow(result)
            plt.show()

        return result

    # find lane 
    # params: processed image
    def lane_search(self, binary_warped):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        if self.show_images:
            # warped image
            f, (ax1,ax2) = plt.subplots(2, 1, figsize=(5,10))
            ax1.imshow(binary_warped, cmap='gray')
            ax1.grid(color='r', linestyle='-', linewidth=2)
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
            ax2.plot(histogram)
            ax2.set_ylabel('White Pixel Count')
            ax2.grid(linestyle='-', linewidth=0.5)
            plt.show()

        midpoint = np.int(histogram.shape[0]//2)
        if True: #not (self.tracking_left or self.enable_tracking):
            leftx_base = np.argmax(histogram[:midpoint])
        else:
            leftx_base = self.leftx_current
        if True: #(self.tracking_left or self.enable_tracking):
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        else:
            rightx_base = self.leftx_current

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            #Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.median(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.median(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # # Generate x and y values for plotting
        self.fity = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        self.tracking_left = left_lane_inds.size > 0
        self.tracking_right  = right_lane_inds.size > 0

        radius_of_curvature = 0.0
        num_of_radii = 0.0

        if self.tracking_left:
            self.left_lane_found = True
            self.leftx_current = leftx_current
            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            self.left_fitx = left_fit[0]*self.fity**2 + left_fit[1]*self.fity + left_fit[2]
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            curv_left = np.mean(self.curvature(self.fity*self.ym_per_pix, self.left_fitx*self.xm_per_pix))
            radius_of_curvature = radius_of_curvature +curv_left
            num_of_radii = num_of_radii + 1.0
        else:
            self.left_fitx = None


        if self.tracking_right > 0:
            self.right_lane_found = True
            self.rightx_current = rightx_current
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
            right_fit = np.polyfit(righty, rightx, 2)
            self.right_fitx = right_fit[0]*self.fity**2 + right_fit[1]*self.fity + right_fit[2]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            curv_right = np.mean(self.curvature(self.fity*self.ym_per_pix, self.right_fitx*self.xm_per_pix))
            radius_of_curvature = radius_of_curvature + curv_right
            num_of_radii = num_of_radii + 1.0
        else:
            self.right_fitx = None 

        if self.tracking_right | self.tracking_left:
            radius_of_curvature = radius_of_curvature / num_of_radii
            self.rad_text = "Radius of Curvature = {}(m)".format(round(radius_of_curvature))


        if self.show_images:
            plt.imshow(out_img)
            if self.tracking_left:
                plt.plot(self.left_fitx, self.fity, color='yellow')
            if self.tracking_right:
                plt.plot(self.right_fitx, self.fity, color='yellow')
            plt.xlim(0, self.img_size[0])
            plt.ylim(self.img_size[1], 0)

        if self.show_images:
            plt.show()

        return out_img

    def curvature(self, y, x):
        # generalised piece wise curvature calculation
        dx  = np.diff(x,1)[1:]
        ddx = np.diff(x,2)
        dy  = np.diff(y,1)[1:]
        ddy = np.diff(y,2)
        x   = x[2:]
        y   = y[2:]

        curvature = ((dx*ddy - dy*ddx)/(dx*dx+dy*dy)**1.5)
        return 1/curvature

    def transform_perspective(self, image_path):

        print ('Transforming perspective to top down..')

        image_original = mpimg.imread(image_path)
        image = cv2.undistort(image_original, self.mtx, self.dist, None, self.mtx)

        height, width, channel = image.shape
        self.img_size = (width, height)

        # Source points
        src = np.float32([[690,450],
                            [1110,self.img_size[1]],
                            [175,self.img_size[1]],
                            [595,450]])

        # destination points to transfer
        offset = 300 # offset for dst points    
        dst = np.float32([[self.img_size[0]-offset, 0],
                          [self.img_size[0]-offset, self.img_size[1]],
                          [offset, self.img_size[1]],
                          [offset, 0]])     
        # # Compute the inverse perspective transform:
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        top_down = cv2.warpPerspective(image, self.M, self.img_size, flags=cv2.INTER_LINEAR)

        if False and self.show_images:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(image)
            ax1.plot(src[:,0], src[:,1], 'r.-')
            ax1.set_title('Undistorted Image')

            ax2.imshow(top_down)
            ax2.plot(dst[:,0], dst[:,1], 'r.-')
            ax2.set_title('Undistorted and Warped Image')
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

        print ('Calibrating camera lens distortion..')

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

        if False and self.show_images:
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

LF = lane_finding(DEBUG, TRACKING)

LF.calibrate(CALIBRATION_FOLDER)
LF.transform_perspective(TEST_FOLDER+'/'+'straight_lines1.jpg')

if DEBUG:
    print ('Process here..')
    image_names = os.listdir(TEST_FOLDER)
    for image_name in image_names:
        if image_name[-4:] == '.jpg': # process images only
            image_path = TEST_FOLDER + '/' + image_name
            image = cv2.imread(image_path)
            LF.process_image(image)
else:
    print('Live video..')
    # white_output = 'test_videos_output/solidWhiteRight.mp4'
    # clip = VideoFileClip("test_videos/solidWhiteRight.mp4")
    # white_clip = clip1.fl_image(process_image)
    # white_clip.write_videofile(white_output, audio=False)



