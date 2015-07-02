# PythonSIFT

This is an implementation of SIFT done entirely in Python with the help of NumPy. A wrapper function, ```match_template()```, matches a template to an image and displays the result as a demonstration of the SIFT algorithm. 

Note: this code relies on OpenCV version 2.4.11.

# Match a template to an image
The wrapper function ```match_template()``` is used to call ```detect_keypoints()```.

Running from python in terminal:
```python
from siftmatch import match_template
match_template(imagename, templatename, threshold, cutoff)
```
where ```imagename``` and ```templatename``` are filename strings (e.g., ```"image.jpg"```), ```threshold``` is the contrast threshold for the sift detector, and ```cutoff``` is the maximum distance between a keypoint descriptor in the image and a keypoint descriptor in the template for the two keypoints to be considered a match. A good value for ```threshold``` is ```5```.

Note that if there are too many keypoints, ```flann.knnSearch()``` on line ```16``` of ```siftmatch.py``` may fail if you don't have enough RAM. Increasing ```threshold``` will reduce the number of keypoints found by SIFT. 

# Use the SIFT detector/descriptor function directly

Running from python in terminal:
```python
from siftdetector import detect_keypoints
[keypoints, descriptors] = detect_keypoints(imagename, threshold)
```
where ```imagename``` and ```threshold``` are defined as above, ```keypoints``` is an ```n``` by ```4``` numpy array that holds the ```n``` keypoints (the first column is the image row coordinate, the second column is the image column coordinate, the third column is the scale, and the fourth column is the orientation as a bin index), and descriptors is an ```n``` by ```128``` numpy array where each row is the SIFT descriptor for the respective keypoint.
