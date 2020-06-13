from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
root = Tk()

class MainWindow():



    def __init__(self, main):

        # canvas for image
        main['width']=600
        main['height']=600

        im = Image.open("C:/Users/user/Downloads/tau.png")

        im = im.resize((200, 200))
        tkimage = ImageTk.PhotoImage(im)
        self.myvar = Label(main,image=tkimage)
        self.myvar.image = tkimage
        self.myvar.grid(row=1,column=1)

        # button to change image
        self.button = Button(main, text="Change", command=self.onButton)
        self.button.grid(row=3,column=1)




    def onButton(self):

        # next image
        path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"),
                                                         ("jpeg files", "*.jpeg")))
        im = Image.open(path)
        #im.show()
        im=im.resize((200,200))
        tkimage = ImageTk.PhotoImage(im)
        self.myvar.configure(image=tkimage)
        self.myvar.image = tkimage
        self.myvar.grid(row=1, column=1)





MainWindow(root)
root.mainloop()