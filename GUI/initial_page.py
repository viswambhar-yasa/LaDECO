# -*- coding: utf-8 -*-
## This file contains graphical user interface module for LaDECO project
### AUTHOR : VISWAMBHAR REDDY YASA
### MATRICULATION NUMBER : 65074
### STUDENT PROJECT TUBF: Projekt LaDECO (Machine learning for lack detection on thermographic videos)

# importing the libraries
import os
from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
from tkinter.ttk import Progressbar
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
### importing data handling class from thermograms module
from thermograms.Data_processing import Data_Handling
# creating a tkinter class
LADECO = Tk()
# assigning the size of the window
width_of_loading_window = 427
height_of_loading_window = 250
width_of_screen = LADECO.winfo_screenwidth()
height_of_screen = LADECO.winfo_screenwidth()
pixcel_along_x = (width_of_screen / 2) - (width_of_loading_window / 2)
pixcel_along_y = (height_of_screen / 2) - (height_of_loading_window / 2)
LADECO.geometry("%dx%d+%d+%d" % (width_of_loading_window, height_of_loading_window, pixcel_along_x, pixcel_along_y))
LADECO.resizable(False, False)
LADECO.attributes('-alpha', 0.9)
LADECO.overrideredirect(1)
a = '#000000'
Frame(LADECO, width=427, height=250, bg=a).place(x=0, y=0)  # 249794
title = Label(LADECO, text='LaDeCO', fg='white', bg=a)
lst1 = ('Calibri (Body)', 18, 'bold')
title.config(font=lst1)
title.place(x=110, y=80)
sub_title = Label(LADECO, text='Laser Coating Detetction and removal', fg='white', bg=a)
lst2 = ('Calibri (Body)', 10)
sub_title.config(font=lst2)
sub_title.place(x=75, y=110)

# main window with contains the menu bar 
def main_window():
    def donothing():
        filewin = Toplevel(main_window)
        button = Button(filewin, text="Do nothing button")
        button.pack()
    # first menu bar operation 
    def data_extraction():
        main_window.destroy()

        def select_directory():
            global root_directory
            root_directory = fd.askdirectory(initialdir='.')
            file_path = os.path.join(root_directory, 'evaluation-list.conf')
            if os.path.isfile(file_path):
                config = open(file_path, 'r')
                file_data = config.read()
                txtarea.insert(END, file_data)
                config.close()
            else:
                file_list = os.listdir(root_directory)
                with open(file_path, 'w') as file_data:
                    for i in file_list[:-1]:
                        file_data.write('#' + i + '\n')
                config = open(file_path, 'r')
                file_data = config.read()
                txtarea.insert(END, file_data)
                config.close()
            pass

        def loading_data():
            try:
                file_path = os.path.join(root_directory, 'evaluation-list.conf')
            except:
                select_directory()
                file_path = os.path.join(root_directory, 'evaluation-list.conf')
            config = open(file_path, 'w')
            data = str(txtarea.get(1.0, END))
            config.write(data)
            config.close()
            output_file_name = simpledialog.askstring("Dialog Box", prompt='Enter file name',
                                                      parent=data_ex)
            print(root_directory)
            file_name = output_file_name + '.hdf5'
            Variantenvergleich = Data_Handling(file_name, root_directory)
            Variantenvergleich.load_data(disp=False)
            pass
        # data extraction module
        data_ex = Tk()
        data_ex.title('DATA EXTRACTION')
        data_ex.geometry('780x420')
        data_ex.resizable(False, False)
        data_ex.iconbitmap(r'./utilites/GUI_utilites/laser_icon.ico')
        frame = Frame(data_ex)
        frame.pack(pady=20)
        # vertical scroll bar to move the text on the screen
        scroll_bar_v = Scrollbar(frame, orient=VERTICAL)
        scroll_bar_v.pack(side=RIGHT, fill=BOTH)

        scroll_bar_h = Scrollbar(frame, orient=HORIZONTAL)
        scroll_bar_h.pack(side=BOTTOM, fill=BOTH)

        txtarea = Text(frame, width=80, height=20)
        txtarea.pack(side=LEFT)

        txtarea.config(yscrollcommand=scroll_bar_v.set)
        scroll_bar_v.config(command=txtarea.yview)

        txtarea.config(xscrollcommand=scroll_bar_h.set)
        scroll_bar_h.config(command=txtarea.xview)
        # adding button for respective operation
        Button(data_ex, text='Open Folder', command=select_directory, border=0, fg=a, bg='white').pack(side=LEFT,
                                                                                                       expand=True,
                                                                                                       fill=X, padx=20)
        Button(data_ex, text="Load data", command=loading_data, border=0, fg=a, bg='white').pack(side=LEFT, expand=True,
                                                                                                 fill=X, padx=20)
        Button(data_ex, text="Exit", command=lambda: data_ex.destroy(), border=0, fg=a, bg='white').pack(side=LEFT,
                                                                                                         expand=True,
                                                                                                         fill=X,
                                                                                                         padx=20)

        data_ex.mainloop()

    pass

    def simulation():
        main_window.destroy()

        def select_directory():
            global root_directory
            root_directory = fd.askdirectory(initialdir='.')
            file_path = os.path.join(root_directory, 'evaluation-list.conf')
            if os.path.isfile(file_path):
                config = open(file_path, 'r')
                file_data = config.read()
                txtarea.insert(END, file_data)
                config.close()
            else:
                file_list = os.listdir(root_directory)
                with open(file_path, 'w') as file_data:
                    for i in file_list[:-1]:
                        file_data.write('#' + i + '\n')
                config = open(file_path, 'r')
                file_data = config.read()
                txtarea.insert(END, file_data)
                config.close()
            pass

        def loading_data():
            try:
                file_path = os.path.join(root_directory, 'evaluation-list.conf')
            except:
                select_directory()
                file_path = os.path.join(root_directory, 'evaluation-list.conf')
            config = open(file_path, 'w')
            data = str(txtarea.get(1.0, END))
            config.write(data)
            config.close()
            output_file_name = simpledialog.askstring("Dialog Box", prompt='Enter file name',
                                                      parent=data_ex)
            print(root_directory)
            file_name = output_file_name + '.hdf5'
            Variantenvergleich = Data_Handling(file_name, root_directory)
            Variantenvergleich.load_data(disp=False)
            pass

        data_ex = Tk()
        data_ex.title('DATA SIMULATION')
        data_ex.geometry('780x420')
        data_ex.resizable(False, False)
        data_ex.iconbitmap(r'./utilites/GUI_utilites/laser_icon.ico')
        frame = Frame(data_ex)
        frame.pack(pady=20)
        scroll_bar_v = Scrollbar(frame, orient=VERTICAL)
        scroll_bar_v.pack(side=RIGHT, fill=BOTH)

        scroll_bar_h = Scrollbar(frame, orient=HORIZONTAL)
        scroll_bar_h.pack(side=BOTTOM, fill=BOTH)

        txtarea = Text(frame, width=80, height=20)
        txtarea.pack(side=LEFT)

        txtarea.config(yscrollcommand=scroll_bar_v.set)
        scroll_bar_v.config(command=txtarea.yview)

        txtarea.config(xscrollcommand=scroll_bar_h.set)
        scroll_bar_h.config(command=txtarea.xview)
        Button(data_ex, text='Open Folder', command=select_directory, border=0, fg=a, bg='white').pack(side=LEFT,
                                                                                                       expand=True,
                                                                                                       fill=X, padx=20)
        Button(data_ex, text="Load data", command=loading_data, border=0, fg=a, bg='white').pack(side=LEFT, expand=True,
                                                                                                 fill=X, padx=20)
        Button(data_ex, text="Exit", command=lambda: data_ex.destroy(), border=0, fg=a, bg='white').pack(side=LEFT,
                                                                                                         expand=True,
                                                                                                         fill=X,
                                                                                                         padx=20)

        data_ex.mainloop()

    pass

    LADECO.destroy()
    main_window = Tk()
    main_window.title('LaDECO')
    main_window.geometry('780x420')
    main_window.resizable(width=False, height=False)
    main_window.iconbitmap(r'./utilites/GUI_utilites/laser_icon.ico')
    menubar = Menu(main_window)
    menubar = Menu(main_window)
    extraction = Menu(menubar, tearoff=0)
    extraction.add_command(label="Extraction", command=data_extraction)
    extraction.add_command(label="Simulation", command=donothing)
    extraction.add_command(label="Save", command=donothing)

    extraction.add_separator()

    extraction.add_command(label="Exit", command=main_window.quit)
    menubar.add_cascade(label="DATA EXTRACTION", menu=extraction)

    main_window.config(menu=menubar)
    main_window.mainloop()
    pass


img = Image.open(r'./utilites/GUI_utilites/laser_icon.png')
img_resize = img.resize((25, 25), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(img_resize)
lab = Label(image=photo).place(x=80, y=80)

LADECO_start = Button(LADECO, width=10, height=1, text='START', command=main_window, border=0, fg=a, bg='white')
LADECO_start.place(x=170, y=200)

LADECO.mainloop()
