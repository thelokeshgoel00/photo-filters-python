from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
root = Tk()
root.title('Image Editor')
root.geometry('500x500')
root.configure(bg='#05E5F9')
root.resizable(False, False)

class MainWindow():



    def __init__(self, main):

        # canvas for image

        im = Image.open("C:/Users/user/Downloads/white.png")

        im = im.resize((250, 250))
        tkimage = ImageTk.PhotoImage(im)
        self.myvar = Label(main,image=tkimage)
        self.myvar.image = tkimage
        self.myvar.place(x=130,y=10)

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
        self.cartooning = Button(main, text="Cartooning", command=self.Cartooning)
        self.cartooning.place(x=370, y=430)





    def onButton(self):

        # next image
        path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"),
                                                         ("jpeg files", "*.jpeg")))
        im = Image.open(path)
        #im.show()
        im=im.resize((250,250))
        tkimage = ImageTk.PhotoImage(im)
        self.myvar.configure(image=tkimage)
        self.myvar.image = tkimage

    def Detect_Face_Mask(self):
        return

    def Gaussian_Blur(self):
        return

    def Convolution_Blur(self):
        return

    def Save_Image(self):

        path = filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("jpeg files", "*.jpeg")))

    def Detect_Face(self):
        return

    def Detect_Edge(self):
        return

    def Cartooning(self):
        return



MainWindow(root)
root.mainloop()