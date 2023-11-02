# Image-Colouring

Requirements
Before running the code, you need to have the following software and data files:

Python 3.x
OpenCV (Open Source Computer Vision Library)
Caffe (Deep learning framework)
Pre-trained Caffe model files:
colorization_deploy_v2.prototxt: Network architecture definition
colorization_release_v2.caffemodel: Pre-trained model with learned parameters
pts_in_hull.npy: Data file used to guide the colorization process
A black and white image you want to colorize
Code Overview
The code can be divided into the following sections:

Importing Libraries and Setting Parameters:

Import necessary Python libraries, including OpenCV and NumPy.
Define parameters such as image dimensions and file paths.
Loading the Caffe Model:

Load the Caffe model from the prototxt and caffemodel files.
Loading the Points in Hull Data:

Load the data from pts_in_hull.npy and prepare it for use in the colorization process.
Loading the Input Image:

Load the black and white image that you want to colorize.
Colorization Loop:

Process the input image and perform colorization in a loop.
Convert the input image to LAB color space.
Resize and preprocess the image for colorization.
Set the L channel as input to the Caffe network and generate AB channel predictions.
Combine the L and colorized AB channels to form a LAB image.
Convert the LAB image back to BGR to obtain the colorized image.
Display the original, grayscale, and colorized images.
Usage
To use the code, follow these steps:

Ensure that you have Python, OpenCV, and Caffe installed on your system.

Download or prepare the required model files and data file (prototxt, caffemodel, and pts_in_hull.npy) and place them in the specified file paths.

Provide the file path to the black and white image you want to colorize in the image_path variable.

Run the code, and it will display the original, grayscale, and colorized versions of the input image in a graphical window.

Conclusion :

This code showcases how to use OpenCV and Caffe to colorize black and white images. By following the provided documentation and making the necessary preparations, you can bring life and color to your monochrome photos.
