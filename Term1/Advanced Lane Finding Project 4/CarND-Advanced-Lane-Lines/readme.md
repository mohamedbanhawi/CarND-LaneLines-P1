**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
This [readme](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/readme.md)

The pipeline for the entire project is [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py)

I created a `lane_finding` class to handle all aspects of the lane finding pipeline.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate` method of the `lane_finding` class in [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Calibrate](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Calibrate_Camera.png "Distortion")

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

This is an example of an undistorted image, the effect is cleary around the corners of the image, notice how the shape of the white car is changed.
![Undistorted](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Undistorted.png "Undistorted")

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in the `process_image` method of the `lane_finding` class in [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py)

An example of the thresholds use is provided here
![Calibrate](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Thresholding.png "Thresholds")

After testing with different threshods I ended up with combining output from the following thesholds:

| Thresholds    | Maximum       | Minimum       | 
|:-------------:|:-------------:|:-------------:| 
| S-Channel     | 150        | 250   | 
| R-Channel     | 200      | 255   | 
| X-Sobel       | 10      | 100   | 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a method called `transform_warp()`method of the `lane_finding` class in [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py).  This function takes as inputs an image (`img`), and sets source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Source and Destination Pts](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Top_View.png "Source and Destination Pts")

Up to this stage, we end up with a thresholded and warped image as follows,

![Warped](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Warped.png "top view")

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

I used a histogram of the bottom half of the image to find left and right lanes to find the base of the lane. This is implemented in lines 157-183 in `lane_search()`method of the `lane_finding` class in [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py).

![Peaks](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Peaks.png "Peaks")

Starting by the base of we divide the base left and right lanes into nine windows. We start from the lowest lane (base lane) and search for detected pixles based on the threholded binary image. If an image exceed a particular number of pixles the following window is recentred to the mean of the pixles in this box. This is implemented in lines 206-230 in `lane_search()`method of the `lane_finding` class in [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py).

![Windows](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/green_mean.png "Windows")

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

A least square quadratic fit was used to generate the lane using the built in `numpy`. The quadtratic fit result is shown below in yellow.

![Lane fit](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/lane_search.png "quadtratic fit")

These values are in the pixel dimensions, they are transformed to metres using the following multipliers:

```python
self.ym_per_pix = 30/720 # meters per pixel in y dimension
self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

I used a generalised curvature calculation to find the curvature for any type of fit in `curvature` method of the `lane_finding` class in [lane_finding.py](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/lane_finding.py).

The resulting curature radius is taken as the average of both lanes.




#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is an example of the output image, this includes the lane area in blue using `cv2.fillPoly` in blue, the left and right lanes are plotted in red `cv2.polylines`. I added the thresholded and lane search results in the top corners to illusrate the lane search pipeline.

The radius of curvature and distance from the lane are written using `cv2.putText`

![Output](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output_images/Output_image.png "Output")

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's the link to my video [Video](https://github.com/mohamedbanhawi/Udacity_SelfDrivingCar_Nanodegree/blob/master/Term1/Advanced%20Lane%20Finding%20Project%204/CarND-Advanced-Lane-Lines/output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
