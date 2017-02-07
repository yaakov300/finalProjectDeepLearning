from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
import tkFileDialog


class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)


        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("Windows")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(1, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        labelfont = ('ariel', 20)
        title = Label(self, text="Object Recognition with Deep Learning")
        title.config(font=labelfont)
        title.grid(row=0, sticky = W, pady=4, padx=5)

        # area = Text(self)
        # area.grid(row=1, column=0, columnspan=2, rowspan=4,
        #           padx=5, sticky=E + W + S + N)

        openFilbtn = Button(self, text="Open File", command=self.onOpen)
        openFilbtn.grid(row=1, column=0, sticky= W + N, padx=10)

        runbtn = Button(self, text="Run")
        runbtn.grid(row=1, column=0, sticky= W + S, padx=10)

        image = Image.open("Monkey-Photos.jpg")
        image = image.resize((370, 330), Image.ANTIALIAS)
        bardejov = ImageTk.PhotoImage(image)
        label1 = Label(self, image=bardejov)
        label1.image = bardejov
        label1.grid(row=1, column=0, padx = 150)

        # bard = Image.open("Monkey-Photos.jpg")
        # bardejov = ImageTk.PhotoImage(bard)
        # label1 = Label(self, image=bardejov)
        # label1.image = bardejov
        # label1.grid(row=1, column=3)
        #x=5, y=5, relwidth=1, relheight=1, width=-10, height=-10


        textTxt = Text(self, height=5, width=73)
        textTxt.insert(END, "Just a text Widget\nin two lines\n")
        textTxt.grid(row=2, column=0, sticky= W + N, padx=5, pady=5)

    def onOpen(self):
        ftypes = [('Python files', '*.py'), ('All files', '*')]
        dlg = tkFileDialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            print("file name = {}".format(fl))
            # text = self.readFile(fl)
            # self.txt.insert(END, text)

    # def readFile(self, filename):
    #     f = open(filename, "r")
    #     text = f.read()
    #     return text


def main():
    root = Tk()
    root.geometry("550x560+150+150")
    app = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()

#
#
# class Example(Frame):
#     def __init__(self, parent):
#         Frame.__init__(self, parent)
#
#         self.parent = parent
#         self.initUI()
#
#     def initUI(self):
#         self.parent.title("DEEP LEARNING")
#         self.pack(fill=BOTH, expand=True)
#
#         frame1 = Frame(self)
#         frame1.pack(fill=X)
#
#         labelfont = ('ariel' ,20)
#         title = Label(frame1, text="Object Recognition with Deep Learning")
#         title.pack(side=BOTTOM, padx=5, pady=5)
#         title.config(font=labelfont)
#
#         frame2 = Frame(self)
#         frame2.pack(fill=X)
#
#         # lbl2 = Label(frame2, text="Author", width=6)
#         # lbl2.pack(side=LEFT, padx=5, pady=5)
#         #
#         # entry2 = Entry(frame2)
#         # entry2.pack(fill=X, padx=5, expand=True)
#
#         frame3 = Frame(self)
#         frame3.pack(fill=BOTH, expand=True)
#
#         lbl3 = Label(frame3, text="Review", width=6)
#         lbl3.pack(side=LEFT, anchor=N, padx=5, pady=5)
#
#         txt = Text(frame3)
#         txt.pack(fill=BOTH, pady=5, padx=5, expand=True)
#
#
# def main():
#     root = Tk()
#     root.geometry("850x500+150+150")
#     app = Example(root)
#     root.mainloop()
#
#
# if __name__ == '__main__':
#     main()