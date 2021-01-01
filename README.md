# PythonSIFT

This is an implementation of SIFT (David G. Lowe's scale-invariant feature transform) done entirely in Python with the help of NumPy. This implementation is based on OpenCV's implementation and returns OpenCV `KeyPoint` objects and descriptors, and so can be used as a drop-in replacement for OpenCV SIFT. This repository is intended to help computer vision enthusiasts learn about the details behind SIFT.

### *Update 2/11/2020*

PythonSIFT has been reimplemented (and greatly improved!) in Python 3. You can find the original Python 2 version in the `legacy` branch. However, I strongly recommend you use `master` (the new Python 3 implementation). It's much better.

## Dependencies

`Python 3`

`NumPy`

`OpenCV-Python`

Last tested successfully using `Python 3.8.5`, `Numpy 1.19.4` and `OpenCV-Python 4.3.0`.

## Usage

```python
import cv2
import pysift

image = cv2.imread('your_image.png', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
```

It's as simple as that. Just like OpenCV.

The returned `keypoints` are a list of OpenCV `KeyPoint` objects, and the corresponding `descriptors` are a list of `128` element NumPy vectors. They can be used just like the objects returned by OpenCV-Python's SIFT `detectAndCompute` member function. Note that this code is not optimized for speed, but rather designed for clarity and ease of understanding, so it will take a few minutes to run on most images.

## Tutorial

You can find a step-by-step, detailed explanation of the code in this repo in my two-part tutorial:

[Implementing SIFT in Python: A Complete Guide (Part 1)](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-1-306a99b50aa5)

[Implementing SIFT in Python: A Complete Guide (Part 2)](https://medium.com/@russmislam/implementing-sift-in-python-a-complete-guide-part-2-c4350274be2b)

I'll walk you through each function, printing and plotting things along the way to develop a solid understanding of SIFT and its implementation details.

## Template Matching Demo

I've adapted OpenCV's SIFT template matching demo to use PythonSIFT instead. The OpenCV images used in the demo are included in this repo for your convenience.
```python
python template_matching_demo.py
```

## Questions, Concerns, Bugs

Anyone is welcome to report and/or fix any bugs. I will resolve any opened issues as soon as possible.

Any questions about the implementation, no matter how simple, are welcome. I will patiently explain my code to you.

### *Original Paper*

["Distinctive Image Features from Scale-Invariant Keypoints", David G. Lowe](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

Definitely worth a read!

### *Legal Notice*

SIFT *was* patented, but it has expired.
This repo is primarily meant for educational purposes, but feel free to use my code any way you want, commercial or otherwise. All I ask is that you cite or share this repo.

You can find the original (now expired) patent [here](https://patents.google.com/patent/US6711293B1/en) (Inventor: David G. Lowe. Assignee: University of British Columbia.).
