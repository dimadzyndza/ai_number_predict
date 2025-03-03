import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import ImageOps, ImageGrab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NumberPredictor(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.initUI()
        self.model = load_model("AI_numb_recognize.h5")

    def initUI(self):
        self.master.title("Number Predictor")
        self.master.geometry("600x400")

        main_frame = Frame(self.master)
        main_frame.pack(fill=BOTH, expand=True)

        self.canvas = Canvas(main_frame, width=280, height=280, bg="#ebd0cc")
        self.canvas.pack(side=LEFT, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.arange(10))
        self.ax.set_ylim([0, 1])
        self.ax.set_xlim([0, 9])
        self.ax.set_xticklabels(np.arange(10))
        self.bar = self.ax.bar(np.arange(10), np.zeros(10))

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_plot.get_tk_widget().pack(side=RIGHT, padx=10, pady=10)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x + 8, y + 8, fill="black", outline="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def get_image_from_canvas(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        return img_array


    def predict_number(self):
        img = self.get_image_from_canvas()
        img = img.reshape(1, 28, 28, 1)  # Підготовка зображення для моделі
        predictions = self.model.predict(img)[0]
        self.update_graph(predictions)

    def update_graph(self, predictions):
        for rect, value in zip(self.bar, predictions):
            rect.set_height(value)
        self.canvas_plot.draw()

if __name__ == '__main__':
    root = Tk()
    app = NumberPredictor(master=root)
    btn_predict = Button(app, text="Передбачити", command=app.predict_number)
    btn_predict.grid(column=0, row=0, columnspan=3)
    btn_clear = Button(app, text="Очистити", command=app.clear_canvas)
    btn_clear.grid(column=4, row=0, columnspan=3)
    app.mainloop()
