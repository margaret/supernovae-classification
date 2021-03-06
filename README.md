Supernovae Classification From Telescope Images
===============================================
Best Group NA's final project for CS 194-16 (Data Science)

About
-----
A supernovae classifier based on http://arxiv.org/pdf/1209.3775v1.pdf

Overview
--------
The volume of data obtained from modern astronomical imaging techniques (e.g. nightly automated scans from large telescopes) is too large for direct human inspection. While some objects are stable and will be visible in many images, others are transient and only visible in some images. Supernovae are in the latter class, and reliable detection of them requires continual processing of nightly images. This project explores automation of the process, using some hand-categorized images as training data. From the training data, the goal is to train automatic classifiers that are as accurate as possible and reasonably fast.

Questions
---------
The primary question is a binary classification "is this image from a supernova"? Using hand-labelled data you will train a classifier and test it using cross-validation.

Data
----
sr_training_77811_stamp21.dat
This file contains raw pixel values for 21x21 pixel postage stamp cutouts of the new, reference, and subtracted images (the origin and purpose of these types of images is described in greater detail in section 1 of the paper). It also has the class labels. The structure of the file is as follows:

Columns:
0) cand_id
1) 1/0 for Real/Bogus (class labels)
2-1324) 3x21x21 numbers for new, reference, and subtracted stamps centered on each object (in that order). The first 441 entries correspond to a np.ravel()'ed version of the new image, the next 441 to a np.ravel()'ed version of the ref image, etc.

The next is samples_features_77811.csv.

It contains the values of features described in the paper. (Section 2.2).

