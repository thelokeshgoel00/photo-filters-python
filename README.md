# Computer Vision Toolkit
This is a GUI based Image Project with a lot of available editing options ranging from simple operations, like bluring of image and cartooning of imange, to some complex operations like face-mask detection, face-detection and edge detection which are implemented through Machine Learning.
## Installation
Please Use Conda Environment on Linux based systems. Then use pip install requirements.txt in the cloned folder. 
Then run gui_frontend.py

## Tasks:-
There are mainly eight different types of tasks(or buttons) available with this editor.
#### 1. Open
We can open an image(only with .jpg or .jpeg extension) through the use of this button. The image can be choosen from any directory. The choosen image is then displayed in front (above the default white screen). All the below operations which you try to apply will all be applied on the image choosen by user and the result of the operation is displayed.

#### 2. Gaussian Blur
Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function. This operation blurs the image by a value specified by the user. On clicking the button it shows up a dialog box which asks user to enter a blur value and blurs the image by that value.

#### 3. Convolution Blur
Convolution is a fundamental operation on images in which a mathematical operation is applied to each pixel to get the desired result. This operation blurs the image by a value specified by the user. On clicking the button it shows up a dialog box which asks user to enter a blur value and blurs the image by that value.

#### 4. Edge Detection
Edges are significant local changes of intensity in a digital image. An edge can be defined as a set of connected pixels that forms a boundary between two disjoint regions. Edge Detection is a method of segmenting an image into regions of discontinuity. Thus the this method detects all the edges in the image and shows them to the user.

#### 5. Face - mask Detection
This is the best feature of our editor that detects the mask on the face of a person. Its accuracy is around 40% and it is a feature programmed by us in the wake of corona pandemic spreading all over the world. Thus it becomes necessary to wear mask in this situation so as to decrease the community spread of this virus. Thus this features detects the mask and shows whether it is present or not on the face.

#### 6. Face Detection
This feature detects the face from the image and even it is able to detect the faces of a group of people in an image. This feature encloses the face inside a rectangular box and shows it to the user.

#### 7. Cartooning
This features enables the cartooning of the image choosen by user and shows the cartooned image back to the user.

#### 8. Save
The last and the important feature of the editor is that it allows the saving of the image to any path choosen by the user. As after applying some operations on the image a user would like to the save the new image and thus it can be done by just clicking the Save button and specifying the path to save the image.


## Tools:-
