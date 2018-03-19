# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline is divide into five stages on a gray scale image.
Gray scale is a requirement for canny edge detection which relies on the gradient.

#### a. Gaussian smoothing
Used a 5x5 matrix as a kernel
This is used to smooth out the gradient false detection due to noise.

#### b. Canny Detection
Edges are identfied by the gradient of the image. 
Thresholds are defined to select edges by removing outliers.

#### C. Masking
This is a heuristic for definign a region of interest, edges outside this region are ignored by setting pixel value to 255.

This was set as a trapezoid, however, this should be adapted for various scenes. The knowledge of the map should inform the shape of the masking.

#### D. Hough Transform
The image is transformed into polar hough space. The parameters for line detection (i.e. threshold for interesction, min length and max gap) were relaxed to include more lines as the masking was useful at removing outliers.

#### E. Lane Fitting
draw_lines() was updated to support two line plotting modes

* Raw: default mode
* line fit mode 

Line fit mode was achieved by assuming left and right lanes would have different slopes. The lines are grouped into different numpy arrays.

For each group of points a line fit is used to estimate a slope.

Knowing the line slope, a point on the line (mean of all points), and the maximum vertical distance for the lane relative to the vehicle (same has mask dimensions) it is possible to extrapolate a line across the lane.



### 2. Identify potential shortcomings with your current pipeline


Identified issues:

* Camera seems to be mounted behind the dashboard. False detection can be trigrred by dirt on windhsield
* Parameters for edge detection are manually set.
* Masking is not adaptable, i.e. the region of interest is fixed, this may fail to detect larger lanes.


### 3. Suggest possible improvements to your pipeline

* Mounting camera on the outside
* Use autothresholding methods, such as http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/ or looking at the histogram of the gradients.
* Using different masks and identifying the mask that results in optimum lane selection per frame.
