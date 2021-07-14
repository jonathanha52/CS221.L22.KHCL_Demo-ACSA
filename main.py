import tkinter as tk
from tkinter.constants import END, W
import torch
import pickle
from tkinter.filedialog import askopenfilename
import os
import numpy as np
import random
from preprocess import data_preprocess 
from load_model import load_model,load_aspect_classifier,load_vectorizer

MODEL_FILE = "D:\PyProject\CapsNet_BERT\MAMS-for-ABSA\Bert_CapsNet.pickle"

root = tk.Tk()
root.title("ACSA Demo")

model = load_model()
model.load_state_dict(torch.load("bert_capsnet.pth"))
vectorizer = load_vectorizer()
aspect_classifier = load_aspect_classifier()

def test():
    #Lay input
    text = txtTestText.get(1.0, END)
    #Trich xuat dac trung va phan loai aspect
    features = vectorizer.transform([text])
    aspects = aspect_classifier.predict(features)
    for i, status in enumerate(aspects[0]):
        varList[i].set(status)
    #Danh gia polarity
    #Duyet qua danh sach varList kiem tra trang thai cac checkbox
    for i, var in enumerate(varList):
        value = var.get()
        txtResultList[i].delete(0, END)
        #Neu checkbox duoc chon, aspect do thuoc input, ta se thuc hien danh gia polarity
        if value == 1:
            aspect = aspect_list[i]
            token, segment = data_preprocess(text, aspect)
            logit = model(token, segment)
            pred = logit.argmax(dim=1)[0].item()
            txtResultList[i].insert(0, polarity_list[pred])
        else:
            txtResultList[i].insert(0, "N/A")
    
polarity_list = ["positive", "negative", "neutral"]
aspect_list = ["food", "service","staff", "price","ambience", "menu", "place", "miscellaneous"]

#varList luu trang thai cua check box
varList = [
    tk.IntVar(),
    tk.IntVar(),
    tk.IntVar(),
    tk.IntVar(),
    tk.IntVar(),
    tk.IntVar(),
    tk.IntVar(),
    tk.IntVar()
]

#Danh sach cac checkbox, checkbox duoc chon tuc la aspect duoc chon cho buoc danh gia polarity
cbOptionList = [
    tk.Checkbutton(root, text="Food", variable=varList[0]),
    tk.Checkbutton(root,text= "Service", variable=varList[1]),
    tk.Checkbutton(root, text="Staff", variable=varList[2]),
    tk.Checkbutton(root, text="Price", variable=varList[3]),
    tk.Checkbutton(root, text="Ambience", variable=varList[4]),
    tk.Checkbutton(root, text="Menu", variable=varList[5]),
    tk.Checkbutton(root, text="Place", variable=varList[6]),
    tk.Checkbutton(root, text="Miscellaneous", variable=varList[7])
]
#Danh sach luu output cua buoc danh gia polarity
txtResultList = [tk.Entry(root),tk.Entry(root),tk.Entry(root),tk.Entry(root),tk.Entry(root),tk.Entry(root),tk.Entry(root),tk.Entry(root)]

#O nhap input
txtTestText = tk.Text(root)

#Nut test
btnTest = tk.Button(root, text="Test", command=test)

#Dua cac widget len frame su dung Grid layout
txtTestText.grid(row=0, column=0, columnspan=2, rowspan=8)
btnTest.grid(row=8, column=0, columnspan=2)

for i, value in enumerate(txtResultList):
    value.grid(row=i, column=3)

for i, value in enumerate(cbOptionList):
    value.grid(row=i, column=2, sticky=W)


tk.mainloop()