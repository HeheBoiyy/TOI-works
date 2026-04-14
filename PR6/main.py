import tkinter as tk
from tkinter import messagebox, font
from app import LetterRecognitionApp

if __name__ == "__main__":
    root = tk.Tk()
    app = LetterRecognitionApp(root)
    root.mainloop()