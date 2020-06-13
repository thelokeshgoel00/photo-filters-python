from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk


class MainWindow():



    def __init__(self, main):

        # canvas for image
        main['width']=800
        main['height']=800
        self.canvas = Canvas(main, width=300, height=300)
        self.canvas.grid(row=0, column=0)

        self.img = PhotoImage(file="tau.png")

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor = NW, image = self.img)

        # button to change image
        self.button = Button(main, text="Change", command=self.onButton)
        self.button.grid(row=1, column=0)



    def onButton(self):

        # next image
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"),
                                                         ("jpeg files", "*.jpeg")))
        filename = "hello.jpg"
        img = ImageTk.PhotoImage(file=filename)

        # change image
        self.canvas.itemconfig(self.image_on_canvas, image = img)



root = Tk()
MainWindow(root)
root.mainloop()