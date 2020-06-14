from tkinter import *
from tkinter import filedialog,messagebox
from PIL import Image,ImageTk
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

root = Tk()
root.title('Image Editor')
root.geometry('500x500')
root.configure(bg='#05E5F9')
root.resizable(False, False)

def convolution(img,img_filter):
    W = img.shape[0]
    H = img.shape[1]
    
    F = img_filter.shape[0]
    new_img = np.zeros((W-F+1,H-F+1))
    
    for row in range(W-F+1):
        for col in range(H-F+1):
            for i in range(F):
                for j in range(F):
                    new_img[row][col] += img[row+i][col+j]*img_filter[i][j]
                    
                if new_img[row][col]>255:
                    new_img[row][col] = 255
                    
                elif new_img[row][col] < 0:
                    new_img[row][col] = 0
    return new_img

class MainWindow():

    def __init__(self, main):

        # canvas for image

        im = Image.open('/home/insanenerd/practice/face_mask/covid-19__face_mask_detection-dataset/COVID-19/training/images/1_Handshaking_Handshaking_1_35.jpg')

        im = im.resize((250, 250))
        tkimage = ImageTk.PhotoImage(im)
        print(type(tkimage))
        self.myvar = Label(main,image=tkimage)
        self.myvar.image = tkimage
        self.myvar.place(x=130,y=10)
        self.saveimg = im
        # button to change image
        self.button = Button(main, text="Choose image", command=self.onButton)
        self.button.place(x=30,y=280)

        # button to detect face mask
        self.face_mask = Button(main, text="Face mask detection", command=self.Detect_Face_Mask)
        self.face_mask.place(x=30,y=330)

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


    def onButton(self):

        # next image
        path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"),
                                                         ("jpeg files", "*.jpeg")))
        im = Image.open(path)
        #im.show()
        im=im.resize((250,250))
        tkimage = ImageTk.PhotoImage(im)
        self.myvar.configure(image=tkimage)
        self.myvar.image = tkimage

    def Detect_Face_Mask(self):
        try:
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            labelfinder = {0:'mask',1:'No Mask'}
            detector = MTCNN()
            try:
                model = load_model('curr_best_model.h5')
            except:
                print("Model Not Found Error")
                messagebox.showerror("Error","Please Add the curr_best_model.h5 file to current directory")
                return self.myvar.image
            try:
                faces = detector.detect_faces(img)
            except:
                print("Face Not detected Error")
                messagebox.showerror("Error","Please check that image is valid jpg type")

                return "File not an jpeg/jpg"
            newimg = img[:]
            #newimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2RGB)
            if len(faces)==0:
                messagebox.showinfo("No Face","No Image Could Be detected in current Image")
            for face in faces:
                curr_box = (face['box'])
                x=curr_box[0]
                y=curr_box[1]
                h=curr_box[2]
                w=curr_box[3]
                im1 = img[y:y+h,x:x+w]
                
                im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
                #print(x,y,h,w)
        #         plt.imshow(im1)
        #         plt.show()
                im1 = cv2.resize(im1,(64,64))
                im1 = np.asarray(im1)
                im1 = preprocess_input(im1)
                a = []
                a.append(im1)
                a = np.array(a)
                pred = model.predict(a)
                predLabel = labelfinder[np.argmax(pred)]
                #print(x,y,w,h)
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                newimg = cv2.rectangle(newimg, (x, y), (x + w, y + h), (color), 4)
                try:
                    cv2.putText(newimg, predLabel, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (color), 1)
                except:
                    cv2.putText(newimg, predLabel, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (color), 1)

            newimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2RGB)
            output = newimg
            pil_image=Image.fromarray(output)
            self.saveimg = pil_image
            tkimage = ImageTk.PhotoImage(pil_image)
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Detect_Face_Mask Error")
            messagebox.showerror("Error","Please run only on RGB images in jpg format")
            return self.myvar.image

    def Gaussian_Blur(self):
        try:
            blur = tkinter.simpledialog.askinteger("askinteger", "Enter blur value")
            print(blur)
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img,(224,224))
            x = 2*blur-1
            output = cv2.GaussianBlur(img,(x,x),cv2.BORDER_DEFAULT)
            pil_image=Image.fromarray(
                              cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                             )
            self.saveimg = pil_image
            tkimage = ImageTk.PhotoImage(pil_image)
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            return tkimage
        except:
            print("Gaussian_Blur Error")
            messagebox.showerror("Error","Please run only on RGB images in jpg format")
            return self.myvar.image

    def Convolution_Blur(self):
        try:
            blur = tkinter.simpledialog.askinteger("askinteger", "Enter blur value")
            print(blur)
            messagebox.showinfo("Loading","Please Wait while updated image is loading")
            x=blur
            blur_filter = np.ones((x,x))/(x*x)
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img,(224,224))
            output = convolution(img,blur_filter)
            pil_image=Image.fromarray(output)
            self.saveimg = pil_image
            tkimage = ImageTk.PhotoImage(pil_image)
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Gaussian_Blur Error")
            messagebox.showerror("Error","Please run only on RGB images in jpg format")
            return self.myvar.image

    def Save_Image(self):
        try:
            path = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("jpeg files", "*.jpeg")))
            self.saveimg.save(path)
        except:
            print("Save_Image Error")
            messagebox.showerror("Error","Please check the file specified again")
            

    def Detect_Face(self):
        try:  
            detector = MTCNN()
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)
            try:
                faces = detector.detect_faces(img)
            except:
                return "File not an jpeg/jpg"
            newimg = img[:]
            for face in faces:
                curr_box = (face['box'])
                x=curr_box[0]
                y=curr_box[1]
                h=curr_box[2]
                w=curr_box[3]
                color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                newimg = cv2.rectangle(newimg, (x, y), (x + w, y + h), (color), 6)
            newimg = cv2.cvtColor(newimg,cv2.COLOR_BGR2RGB)
            output = newimg
            pil_image=Image.fromarray(output)
            self.saveimg = pil_image
            tkimage = ImageTk.PhotoImage(pil_image)
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Detect_Face Error")
            messagebox.showerror("Error","Please run only on RGB images in jpg format")
            return self.myvar.image

    def Detect_Edge(self):
        try:
            img = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR)    
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            img = cv2.fastNlMeansDenoising(img)
            img = cv2.resize(img,(224,224))
            #output2 = convolution(img,edge_filter)
            output = cv2.Canny(img,70,150)
            pil_image=Image.fromarray(output)
            self.saveimg = pil_image
            tkimage = ImageTk.PhotoImage(pil_image)
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Detect_Edge Error")
            messagebox.showerror("Error","Please run only on RGB images in jpg format")
            return self.myvar.image

    def Cartooning(self):
        try:
            img_rgb = cv2.cvtColor(np.array(self.saveimg), cv2.COLOR_RGB2BGR) 
            num_down = 2 # number of downsampling steps
            num_bilateral = 7 # number of bilateral filtering steps
            img_color = img_rgb
            for _ in range(num_down):
                img_color = cv2.pyrDown(img_color)
            for _ in range(num_bilateral):
                img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

            for _ in range(num_down):
                img_color = cv2.pyrUp(img_color)

            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)

            img_edge = cv2.adaptiveThreshold(img_blur, 255,
               cv2.ADAPTIVE_THRESH_MEAN_C,
               cv2.THRESH_BINARY,
               blockSize=9,
               C=2)

            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            output = cv2.bitwise_and(img_color, img_edge)
            pil_image=Image.fromarray(output)
            self.saveimg = pil_image
            tkimage = ImageTk.PhotoImage(pil_image)
            self.myvar.configure(image=tkimage)
            self.myvar.image = tkimage
            print("Image Updated")
            return tkimage
        except:
            print("Cartoonify Error")
            messagebox.showerror("Error","Please run only on RGB images in jpg format")
            return self.myvar.image



MainWindow(root)
root.mainloop()
