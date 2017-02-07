from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
import tkFileDialog
import ImageRecognition


class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)


        self.parent = parent
        self.initUI()

    def initUI(self):
        self.filename = "Monkey-Photos.jpg"
        self.parent.title("Windows")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(1, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        labelfont = ('ariel', 20)
        title = Label(self, text="Object Recognition with Deep Learning")
        title.config(font=labelfont)
        title.grid(row=0, sticky = W, pady=4, padx=60)

        openFilbtn = Button(self, text="Open File", command=self.onOpen)
        openFilbtn.grid(row=1, column=0, sticky= W + N, padx=10)

        runbtn = Button(self, text="Run", command=self.run)
        runbtn.grid(row=1, column=0, sticky= W + S, padx=10)

        image = Image.open("Monkey-Photos.jpg")
        image = image.resize((370, 330), Image.ANTIALIAS)
        bardejov = ImageTk.PhotoImage(image)
        self.imglabel = Label(self, image=bardejov)
        self.imglabel.image = bardejov
        self.imglabel.grid(row=1, column=0, padx = 150)

        self.textTxt = Text(self, height=8, width=90)
        # self.textTxt.insert()
        self.textTxt.grid(row=2, column=0, sticky= W + N, padx=5, pady=5)

    def onOpen(self):
        ftypes = [('Image files', '*.jpg'), ('All files', '*')]
        dlg = tkFileDialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            self.filename = fl
            print("--------fl-------{}".format(fl))
            image = Image.open(fl)
            image = image.resize((370, 330), Image.ANTIALIAS)
            bardejov1 = ImageTk.PhotoImage(image)
            self.imglabel = Label(self, image=bardejov1)
            self.imglabel.image = bardejov1
            self.imglabel.grid(row=1, column=0, padx=150)

    def run(self):
        self.textTxt.delete('1.0', END)
        text = ImageRecognition.cheack_image(self.filename)
        self.textTxt.insert(END, text)


def main():
    root = Tk()
    root.geometry("650x510+150+150")
    app = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()
