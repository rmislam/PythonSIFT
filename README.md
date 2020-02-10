# PythonSIFT

This is an implementation of SIFT (David Lowe's scale-invariant feature transform) done entirely in Python with the help of NumPy. This implementation is based on OpenCV's implementation and returns OpenCV KeyPoint objects and descriptors, and so can be used as a drop-in replacement for OpenCV SIFT. This repository is intended to help computer vision enthusiasts learn about the details behind SIFT.

### *Update 2/11/2020*

PythonSIFT has been reimplemented (and greatly improved!) in Python 3. You can find the original Python 2 version in the `legacy` branch. However, I strongly recommend you use `master` (the new Python 3 implementation). It's much better.

## Dependencies

Python 3
NumPy
OpenCV-Python (`import cv2`)

Last tested successfully using Python 3.7.6 and OpenCV-Python 4.2.0
Note: this code relies on OpenCV version 2.4.11.

## Usage

```python
import cv2
import pysift

image = cv2.imread('your_image.png', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
```

It's as simple as that. Just like OpenCV.

The returned `keypoints` are a list of OpenCV `KeyPoint` objects, and the corresponding `descriptors` are a list of `128` element NumPy vectors. They can be used just like the objects returned by OpenCV-Python's SIFT `detectAndCompute` member function.

## Template Matching Demo

I've adapted OpenCV's SIFT template matching demo to use PythonSIFT instead. The OpenCV images used in the demo are included in this repo for your convenience.
```python
python template_matching_demo.py
```

## Questions, Concerns, Bugs

Anyone is welcome to report and/or fix any bugs. I will resolve any opened issues as soon as possible. Also, any questions about the implementation, no matter how simple you may think they are, are welcome. I will patiently explain my code to you.
