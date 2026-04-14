import tkinter as tk
from tkinter import messagebox, ttk
import json
import os
import sys

class LetterRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание букв (5×5) - XOR сравнение")
        self.root.geometry("700x650")
        self.root.resizable(False, False)

        self.json_filename = "letters.json"
        self.reference = self.load_references()
        if not self.reference:
            messagebox.showerror("Ошибка",
                f"Файл '{self.json_filename}' не найден или пуст.\n"
                "Поместите корректный JSON-файл с эталонами в папку с программой.")
            self.root.destroy()
            sys.exit(1)

        self.user_matrix = [[0 for _ in range(5)] for _ in range(5)]
        self.cell_size = 60
        self.grid_width = 5 * self.cell_size
        self.grid_height = 5 * self.cell_size

        self.create_widgets()
        self.draw_grid()

    def load_references(self):
        if not os.path.exists(self.json_filename):
            return None
        try:
            with open(self.json_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for letter, matrix in data.items():
                if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
                    raise ValueError(f"Неверный формат матрицы для буквы {letter}")
            return data
        except Exception as e:
            messagebox.showerror("Ошибка загрузки JSON", str(e))
            return None

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(main_frame, text="Нарисуйте букву в сетке 5×5 (клик по ячейке)",
                               font=("Arial", 12, "bold"))
        title_label.pack(pady=5)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(pady=5)

        self.canvas = tk.Canvas(canvas_frame, width=self.grid_width, height=self.grid_height,
                                bg='white', highlightthickness=1, highlightbackground='gray')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.clear_btn = tk.Button(button_frame, text="Очистить", command=self.clear_grid,
                                   width=12, bg="#f0f0f0")
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.recognize_btn = tk.Button(button_frame, text="Распознать", command=self.recognize,
                                       width=12, bg="#4CAF50", fg="white")
        self.recognize_btn.pack(side=tk.LEFT, padx=5)

        self.show_ref_btn = tk.Button(button_frame, text="Показать эталоны", command=self.show_references,
                                      width=15, bg="#2196F3", fg="white")
        self.show_ref_btn.pack(side=tk.LEFT, padx=5)

        result_frame = tk.LabelFrame(main_frame, text="Результаты распознавания", padx=10, pady=5)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_text = tk.Text(result_frame, height=6, width=60, state=tk.DISABLED,
                                   font=("Courier New", 10))
        self.result_text.pack(pady=5)

        self.verdict_label = tk.Label(result_frame, text="", font=("Arial", 12, "bold"), fg="#333")
        self.verdict_label.pack(pady=5)

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(5):
            for j in range(5):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "black" if self.user_matrix[i][j] == 1 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray", width=1)

    def on_canvas_click(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= row < 5 and 0 <= col < 5:
            self.user_matrix[row][col] = 1 - self.user_matrix[row][col]
            self.draw_grid()
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.config(state=tk.DISABLED)
            self.verdict_label.config(text="")

    def clear_grid(self):
        self.user_matrix = [[0 for _ in range(5)] for _ in range(5)]
        self.draw_grid()
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.verdict_label.config(text="")

    def xor_distance(self, mat1, mat2):
        distance = 0
        for i in range(5):
            for j in range(5):
                if mat1[i][j] != mat2[i][j]:
                    distance += 1
        return distance

    def recognize(self):
        if not self.reference:
            messagebox.showwarning("Нет эталонов", "Список эталонов пуст.")
            return

        distances = {}
        for letter, ref_mat in self.reference.items():
            d = self.xor_distance(self.user_matrix, ref_mat)
            distances[letter] = d

        min_dist = min(distances.values())
        best_letters = [let for let, d in distances.items() if d == min_dist]

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        # ИСПРАВЛЕННАЯ СТРОКА ↓
        result_lines = [f"{letter}: {distances[letter]} отличий" for letter in sorted(distances.keys())]

        self.result_text.insert(tk.END, "\n".join(result_lines))
        self.result_text.config(state=tk.DISABLED)

        if len(best_letters) == 1:
            verdict = f"Распознана буква: {best_letters[0]} (отличий: {min_dist})"
        else:
            verdict = f"Неоднозначность: {', '.join(best_letters)} (отличий: {min_dist})"
        self.verdict_label.config(text=verdict)

    def show_references(self):
        ref_window = tk.Toplevel(self.root)
        ref_window.title("Эталоны букв (5×5)")
        ref_window.geometry("700x500")
        ref_window.resizable(True, True)

        tk.Label(ref_window, text="Эталонные матрицы букв (чёрный = 1, белый = 0)",
                font=("Arial", 10)).pack(pady=5)

        canvas = tk.Canvas(ref_window, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(ref_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        letters = list(self.reference.keys())
        cols = 4
        for idx, letter in enumerate(sorted(letters)):
            row = idx // cols
            col = idx % cols

            frame = tk.LabelFrame(scrollable_frame, text=f"Буква {letter}", padx=5, pady=5)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            cell_size = 30
            w = 5 * cell_size
            h = 5 * cell_size
            letter_canvas = tk.Canvas(frame, width=w, height=h, bg='white', highlightthickness=0)
            letter_canvas.pack()

            matrix = self.reference[letter]
            for i in range(5):
                for j in range(5):
                    x1 = j * cell_size
                    y1 = i * cell_size
                    x2 = x1 + cell_size
                    y2 = y1 + cell_size
                    color = "black" if matrix[i][j] == 1 else "white"
                    letter_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

        for c in range(cols):
            scrollable_frame.columnconfigure(c, weight=1)

        btn_frame = tk.Frame(ref_window)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Закрыть", command=ref_window.destroy, width=15).pack()

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
