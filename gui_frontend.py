from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tkinter.simpledialog
import os
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random

# uncomment the below 2 lines if using a GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# creating main mindow of gui
root = Tk()

# setting the title of the gui
root.title('Image Editor')

# size of the gui screen
root.geometry('500x500')

# background color of the gui body
root.configure(bg='#05E5F9')

# not allowing the gui to be resizable.
root.resizable(False, False)


def convolution(img, img_filter):
    # width of image
    W = img.shape[0]

    #height of image
    H = img.shape[1]
    F = img_filter.shape[0]
    # numpy matrix initialised with all zeros
    new_img = np.zeros((W - F + 1, H - F + 1))

    for row in range(W - F + 1):
        for col in range(H - F + 1):
            for i in range(F):
                for j in range(F):
                    new_img[row][col] += img[row + i][col + j] * img_filter[i][j]

                if new_img[row][col] > 255:
                    new_img[row][col] = 255

                elif new_img[row][col] < 0:
                    new_img[row][col] = 0
    return new_img


class MainWindow():

    def __init__(self, main):

        # the default image in gui
        im = Image.open(
            '/home/insanenerd/practice/face_mask/covid-19__face_mask_detection-dataset/COVID-19/training/images/1_Handshaking_Handshaking_1_35.jpg')

        #resizing the image
        im = im.resize((250, 250))

        # changing it to ImageTk format to open it in gui
        tkimage = ImageTk.PhotoImage(im)
        print(type(tkimage))

        # label to show the image
        self.myvar = Label(main, image=tkimage)
        self.myvar.image = tkimage
        self.myvar.place(x=130, y=10)
        self.saveimg = im

        # button to change image
        self.button = Button(main, text="Choose image", command=self.onButton)
        self.button.place(x=30, y=280)

        # button to detect face mask
        self.face_mask = Button(main, text="Face mask detection", command=self.Detect_Face_Mask)
        self.face_mask.place(x=30, y=330)

        # button for gaussian blur
        self.gaussian_blur = Button(main, text="Gaussian Blur", command=self.Gaussian_Blur)
        self.gaussian_blur.place(x=30, y=380)

        # button for convolution blur
        self.convolution_blur = Button(main, text="Convolution Blur", command=self.Convolution_Blur)
        self.convolution_blur.place(x=30, y=430)

        # button to save image
        self.save_image = Button(main, text="Save image", command=self.Save_Image)
        self.save_image.place(x=370, y=280)

        # button to detect face
        self.detect_face = Button(main, text="Face detection", command=self.Detect_Face)
        self.detect_face.place(x=370, y=330)

        # button for edge detection
        self.detect_edge = Button(main, text="Edge Detection", command=self.Detect_Edge)
        self.detect_edge.place(x=370, y=380)

        # button for cartooning of image
        self.cartooning = Button(main, text="Cartoonify", command=self.Cartooning)
        self.cartooning.place(x=370, y=430)

    # function for changing the image
    def onButton(self):

        # next image
        path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpg"),
                                                     ("jpeg files", "*.jpeg")))
        im = Image.open(path)

        # resizing the image
        im = im.resize((250, 250))
        self.saveimg = im

        # converting to ImageTk format
        tkimage = ImageTk.PhotoImage(im)

        #changing the image inside the label
        self.myvar.configure(image=tkimage)
        self.myvar.image = tkimage

    # functio for face mask detection
    def Detect_Face_Mask(self):
        try:
            # message box requesting the user to wait because it takes some amount of time to detect the face mask
            messagebox.showinfo("Loading", "Please Wait while updated image is loading")

            # converting RGB to BGR
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)

            # labels for mask or no mask
            labelfinder = {0: 'mask', 1: 'No Mask'}
            detector = MTCNN()
            try:
                # loading the model
                model = load_model('curr_best_model.h5')
            except:
                # if the model is not present inside cuurent directory.
                print("Model Not Found Error")
                messagebox.showerror("Error", "Please Add the curr_best_model.h5 file to current directory")
                return self.myvar.image
            try:
                # firstly detecting faces inside an image
                faces = detector.detect_faces(img)
            except:
                # in case no face is present  inside an image
                print("Face Not detected Error")
                messagebox.showerror("Error", "Please check that image is valid jpg type")

                return "File not an jpeg/jpg"

            newimg = img[:]
            # newimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2RGB)
            if len(faces) == 0:
                messagebox.showinfo("No Face", "No Image Could Be detected in current Image")

            # detecting the mask for every face detected inside the image.
            for face in faces:
                curr_box = (face['box'])
                # top left coordinate of box
                x = curr_box[0]
                y = curr_box[1]
                # height and width of the box
                h = curr_box[2]
                w = curr_box[3]

                # cropping the image
                im1 = img[y:y + h, x:x + w]
                # converting color BGR to RGB
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                # print(x,y,h,w)
                #         plt.imshow(im1)
                #         plt.show()
                # resizing the image
                im1 = cv2.resize(im1, (64, 64))
                # converting it to array
                im1 = np.asarray(im1)
                im1 = preprocess_input(im1)
                a = []
                a.append(im1)
                # creating a numpy array
                a = np.array(a)

                # predicting the mask using our model
                pred = model.predict(a)
                # reading the label
                predLabel = labelfinder[np.argmax(pred)]
                # print(x,y,w,h)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # enclosing inside a rectangle
                newimg = cv2.rectangle(newimg, (x, y), (x + w, y + h), (color), 4)
                try:
                    # put the result on the image as a text
                    cv2.putText(newimg, predLabel, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (color), 1)
                except:
                    cv2.putText(newimg, predLabel, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (color), 1)

            newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
            output = newimg
            # converting array to PIL Image format
            pil_image = Image.fromarray(output)
            self.saveimg = pil_image
            # converting PIL Image to ImageTk format to show inside the label
            tkimage = ImageTk.PhotoImage(pil_image)
            # updating the image inside label
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Detect_Face_Mask Error")
            messagebox.showerror("Error", "Please run only on RGB images in jpg format")
            return self.myvar.image

    # function for gaussian bluring of image
    def Gaussian_Blur(self):
        try:
            # asking the user for blur value
            blur = tkinter.simpledialog.askinteger("askinteger", "Enter blur value")
            print(blur)
            # convert color from RGB to BGR
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            # resizing of image
            img = cv2.resize(img, (224, 224))
            x = 2 * blur - 1
            # creating the gaussian blur
            output = cv2.GaussianBlur(img, (x, x), cv2.BORDER_DEFAULT)
            # creating PIL Image from array
            pil_image = Image.fromarray(
                cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            )
            self.saveimg = pil_image
            # converting PIL Image to ImageTk format to show inside tkinter
            tkimage = ImageTk.PhotoImage(pil_image)
            # updating the image inside label
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            return tkimage
        except:
            print("Gaussian_Blur Error")
            messagebox.showerror("Error", "Please run only on RGB images in jpg format")
            return self.myvar.image

    # function for convolution bluring of image
    def Convolution_Blur(self):
        try:
            # asking the user for blur value.
            blur = tkinter.simpledialog.askinteger("askinteger", "Enter blur value")
            print(blur)
            # requesting the user to wait as loading image might take some time
            messagebox.showinfo("Loading", "Please Wait while updated image is loading")
            x = blur
            blur_filter = np.ones((x, x)) / (x * x)
            # convert color from RBG to BGR
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # craeting a GRAY image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # resizing of image
            img = cv2.resize(img, (224, 224))
            # the final blurred image
            output = convolution(img, blur_filter)

            # converting to PIL image format
            gray_image = Image.fromarray(output)
            pil_image = Image.new("RGB", gray_image.size)
            pil_image.paste(gray_image)
            self.saveimg = pil_image
            # converting to ImageTk format to show inside tkinter
            tkimage = ImageTk.PhotoImage(pil_image)
            # updating the image inside label
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Convolution_Blur Error")
            messagebox.showerror("Error", "Please run only on RGB images in jpg format")
            return self.myvar.image

    def Save_Image(self):
        try:
            # choosing path to save the new image after some applied operations
            path = filedialog.asksaveasfilename(initialdir="/", title="Select file",
                                                filetypes=(("jpeg files", "*.jpg"), ("jpeg files", "*.jpeg")))
            self.saveimg.save(path)
        except:
            print("Save_Image Error")
            messagebox.showerror("Error", "Please check the file specified again")

    # function to detect faces
    def Detect_Face(self):
        try:
            # reuqesting the user to wait until loading of image
            messagebox.showinfo("Loading", "Please Wait while updated image is loading")
            detector = MTCNN()
            # convert color from RBG to BGR
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            try:
                # detecting all the faces inside an image
                faces = detector.detect_faces(img)
            except:
                return "File not an jpeg/jpg"
            newimg = img[:]
            # traversing all the faces
            for face in faces:
                curr_box = (face['box'])
                # top left corodinate
                x = curr_box[0]
                y = curr_box[1]
                # width and height of box
                h = curr_box[2]
                w = curr_box[3]
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # enclosing faces inside a rectangle
                newimg = cv2.rectangle(newimg, (x, y), (x + w, y + h), (color), 6)

            newimg = cv2.cvtColor(newimg, cv2.COLOR_BGR2RGB)
            output = newimg

            # converting image to PIL format
            pil_image = Image.fromarray(output)
            self.saveimg = pil_image

            # converting PIl Image to ImageTk format to show inside tkinter
            tkimage = ImageTk.PhotoImage(pil_image)
            # updating image inside the label
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Detect_Face Error")
            messagebox.showerror("Error", "Please run only on RGB images in jpg format")
            return self.myvar.image

    # function to detect edges in the image
    def Detect_Edge(self):
        try:
            # converting color formats
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # denoising the image
            img = cv2.fastNlMeansDenoising(img)
            # resizing of image
            img = cv2.resize(img, (224, 224))
            # output2 = convolution(img,edge_filter)

            output = cv2.Canny(img, 70, 150)
            # converting the output to PIL format
            pil_image = Image.fromarray(output)
            self.saveimg = pil_image

            # converting the PIL Image to ImageTk format to show inside tkinetr
            tkimage = ImageTk.PhotoImage(pil_image)
            # updating the image inside label
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Detect_Edge Error")
            messagebox.showerror("Error", "Please run only on RGB images in jpg format")
            return self.myvar.image

    # function for cartooning of image
    def Cartooning(self):
        try:
            # convert color formats
            img_rgb = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            # resizing of image
            img_rgb = cv2.resize(img_rgb, (252, 252))
            num_down = 2  # number of downsampling steps
            num_bilateral = 7  # number of bilateral filtering steps
            img_color = img_rgb
            for _ in range(num_down):
                img_color = cv2.pyrDown(img_color)
            for _ in range(num_bilateral):
                # applying bilateral filter on the image
                img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

            for _ in range(num_down):
                img_color = cv2.pyrUp(img_color)

            # changing color format
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)

            img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY,
                                             blockSize=9,
                                             C=2)

            # changing color format
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            print(img_edge.shape)
            print(img_color.shape)

            output = cv2.bitwise_and(img_color, img_edge)
            # converting array to PIL Image
            pil_image = Image.fromarray(output)
            self.saveimg = pil_image

            # converting PIL Image to ImageTk format to show inside tkinter
            tkimage = ImageTk.PhotoImage(pil_image)
            # updating the image inside label
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Cartoonify Error")
            messagebox.showerror("Error", "Please run only on RGB images in jpg format")
            return self.myvar.image


MainWindow(root)
# running the event loop
root.mainloop()
