import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
import os


class ConvolutionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ручная фильтрация изображения (ядро 3×3)")
        self.root.geometry("1000x750")
        self.root.resizable(True, True)

        # Переменные
        self.original_image = None
        self.filtered_image = None
        self.kernel = None
        self.photo_original = None
        self.photo_filtered = None

        # Словарь с предустановленными масками
        self.preset_kernels = {
            "Выберите фильтр...": "",
            "Box Blur (размытие)": "1 1 1 1 1 1 1 1 1",
            "Gaussian Blur (мягкое размытие)": "1 2 1 2 4 2 1 2 1",
            "Sharpen (повышение резкости)": "0 -1 0 -1 5 -1 0 -1 0",
            "Strong Sharpen (сильная резкость)": "-1 -1 -1 -1 9 -1 -1 -1 -1",
            "Laplacian (выделение границ)": "0 -1 0 -1 4 -1 0 -1 0",
            "Laplacian Diagonal (все границы)": "-1 -1 -1 -1 8 -1 -1 -1 -1",
            "Sobel X (вертикальные границы)": "-1 0 1 -2 0 2 -1 0 1",
            "Sobel Y (горизонтальные границы)": "-1 -2 -1 0 0 0 1 2 1",
            "Emboss (тиснение СЗ)": "-2 -1 0 -1 1 1 0 1 2",
            "Emboss East (тиснение Восток)": "-1 0 1 -1 1 1 -1 0 1",
            "Motion Blur X (смаз по горизонтали)": "0 0 0 1 1 1 0 0 0",
            "Motion Blur Y (смаз по вертикали)": "0 1 0 0 1 0 0 1 0",
            "Shift Right (сдвиг вправо)": "0 0 0 0 0 1 0 0 0",
        }

        # Рекомендации по нормировке для каждой маски
        self.normalize_hints = {
            "Box Blur (размытие)": True,
            "Gaussian Blur (мягкое размытие)": True,
            "Sharpen (повышение резкости)": False,
            "Strong Sharpen (сильная резкость)": False,
            "Laplacian (выделение границ)": False,
            "Laplacian Diagonal (все границы)": False,
            "Sobel X (вертикальные границы)": False,
            "Sobel Y (горизонтальные границы)": False,
            "Emboss (тиснение СЗ)": False,
            "Emboss East (тиснение Восток)": False,
            "Motion Blur X (смаз по горизонтали)": True,
            "Motion Blur Y (смаз по вертикали)": True,
            "Shift Right (сдвиг вправо)": False,
        }

        self.create_widgets()

    def create_widgets(self):
        # Верхняя панель управления
        control_frame = tk.Frame(self.root, padx=5, pady=5)
        control_frame.pack(fill=tk.X)

        # === Секция выбора предустановленной маски ===
        preset_frame = tk.LabelFrame(control_frame, text="Предустановленные фильтры", padx=5, pady=5)
        preset_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=5, pady=5)

        tk.Label(preset_frame, text="Выберите фильтр:").pack(side=tk.LEFT, padx=5)

        self.preset_var = tk.StringVar(value="Выберите фильтр...")
        self.preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_var,
            values=list(self.preset_kernels.keys()),
            state="readonly",
            width=35
        )
        self.preset_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.preset_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)

        # Информация о фильтре
        self.filter_info_label = tk.Label(preset_frame, text="", fg="blue", font=("Arial", 9))
        self.filter_info_label.pack(side=tk.LEFT, padx=10)

        # === Ручной ввод маски ===
        kernel_frame = tk.LabelFrame(control_frame, text="Ручной ввод маски 3×3", padx=5, pady=5)
        kernel_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5, pady=5)

        tk.Label(kernel_frame, text="9 чисел через пробел:").pack(side=tk.LEFT, padx=5)

        self.kernel_entry = tk.Entry(kernel_frame, width=40)
        self.kernel_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.kernel_entry.insert(0, "0 -1 0 -1 5 -1 0 -1 0")

        # === Кнопки управления ===
        btn_frame = tk.Frame(control_frame)
        btn_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=5, pady=5)

        self.load_btn = tk.Button(btn_frame, text="📁 Загрузить изображение", command=self.load_image, width=20)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(btn_frame, text="Файл не выбран", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # === Флажок нормировки с подсказкой ===
        norm_frame = tk.Frame(control_frame)
        norm_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        self.normalize_var = tk.BooleanVar(value=False)
        self.normalize_check = tk.Checkbutton(
            norm_frame,
            text="Нормировать на сумму ядра",
            variable=self.normalize_var
        )
        self.normalize_check.pack(side=tk.LEFT)

        self.norm_hint_label = tk.Label(norm_frame, text="", fg="green", font=("Arial", 9))
        self.norm_hint_label.pack(side=tk.LEFT, padx=10)

        # === Кнопки действий ===
        action_frame = tk.Frame(control_frame)
        action_frame.grid(row=3, column=2, columnspan=2, sticky="e", padx=5, pady=5)

        self.apply_btn = tk.Button(action_frame, text="▶ Применить фильтр", command=self.apply_filter,
                                   state=tk.DISABLED, bg="#4CAF50", fg="white", width=18)
        self.apply_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(action_frame, text="💾 Сохранить результат", command=self.save_image,
                                  state=tk.DISABLED, bg="#2196F3", fg="white", width=18)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Прогресс-бар
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        self.progress.grid_remove()

        # === Рамка для изображений ===
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - исходное изображение
        left_frame = tk.LabelFrame(image_frame, text="Исходное изображение")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_canvas = tk.Canvas(left_frame, bg='#f0f0f0', width=400, height=400)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Правая панель - обработанное изображение
        right_frame = tk.LabelFrame(image_frame, text="Результат фильтрации")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.filtered_canvas = tk.Canvas(right_frame, bg='#f0f0f0', width=400, height=400)
        self.filtered_canvas.pack(fill=tk.BOTH, expand=True)

        # Привязка события изменения размера холстов
        self.original_canvas.bind("<Configure>", self.on_resize)
        self.filtered_canvas.bind("<Configure>", self.on_resize)

    def on_preset_selected(self, event=None):
        """Обработчик выбора предустановленного фильтра"""
        selected = self.preset_var.get()

        if selected == "Выберите фильтр...":
            return

        # Получаем маску
        kernel_str = self.preset_kernels.get(selected, "")
        if kernel_str:
            self.kernel_entry.delete(0, tk.END)
            self.kernel_entry.insert(0, kernel_str)

            # Автоматически устанавливаем рекомендованную нормировку
            recommended = self.normalize_hints.get(selected, False)
            self.normalize_var.set(recommended)

            # Обновляем подсказки
            if recommended:
                self.norm_hint_label.config(text="✓ Рекомендуется включить нормировку", fg="green")
            else:
                self.norm_hint_label.config(text="✗ Нормировка не требуется (сумма ≈ 0 или 1)", fg="orange")

            # Показываем сумму ядра
            try:
                numbers = list(map(float, kernel_str.split()))
                kernel_sum = sum(numbers)
                sum_info = f"Сумма ядра: {kernel_sum:.2f}"
                if abs(kernel_sum) < 0.001:
                    sum_info += " (граничный фильтр)"
                elif kernel_sum > 1.1:
                    sum_info += " (усиливающий фильтр)"
                elif kernel_sum < 0.9:
                    sum_info += " (ослабляющий фильтр)"
                else:
                    sum_info += " (нейтральный фильтр)"
                self.filter_info_label.config(text=sum_info)
            except:
                self.filter_info_label.config(text="")

    def load_image_with_pil(self, file_path):
        """Загрузка изображения через PIL"""
        try:
            pil_img = Image.open(file_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            return np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Ошибка загрузки через PIL: {str(e)}")

    def load_image(self):
        """Загрузка изображения через диалог"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            try:
                self.original_image = self.load_image_with_pil(file_path)
            except:
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Не удалось прочитать файл")
                self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.file_label.config(text=os.path.basename(file_path), fg="black")
            self.display_image(self.original_image, self.original_canvas, "original")
            self.apply_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить изображение:\n{str(e)}")
            self.original_image = None
            self.file_label.config(text="Ошибка загрузки", fg="red")
            self.apply_btn.config(state=tk.DISABLED)

    def display_image(self, img, canvas, tag):
        """Масштабирование и отображение изображения на холсте"""
        if img is None:
            canvas.delete("all")
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = img.shape[1]
            canvas_height = img.shape[0]

        img_h, img_w = img.shape[:2]
        scale = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        if tag == "original":
            self.photo_original = photo
        else:
            self.photo_filtered = photo

        canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)

    def on_resize(self, event):
        """Обработчик изменения размера холста"""
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas, "original")
        if self.filtered_image is not None:
            self.display_image(self.filtered_image, self.filtered_canvas, "filtered")

    def manual_convolution(self, image, kernel, normalize):
        """Ручная свёртка с ядром 3×3"""
        h, w, c = image.shape
        img_float = image.astype(np.float64)
        result = np.zeros_like(img_float)

        kernel_sum = np.sum(kernel)
        if normalize and abs(kernel_sum) < 1e-12:
            normalize = False

        for y in range(h):
            for x in range(w):
                for ch in range(c):
                    acc = 0.0
                    for ky in range(-1, 2):
                        for kx in range(-1, 2):
                            ny = y + ky
                            nx = x + kx
                            if ny < 0:   ny = 0
                            if ny >= h:  ny = h - 1
                            if nx < 0:   nx = 0
                            if nx >= w:  nx = w - 1

                            pixel_val = img_float[ny, nx, ch]
                            kernel_val = kernel[ky + 1, kx + 1]
                            acc += pixel_val * kernel_val

                    if normalize:
                        acc /= kernel_sum

                    if acc < 0:
                        acc = 0
                    elif acc > 255:
                        acc = 255

                    result[y, x, ch] = acc

        return result.astype(np.uint8)

    def apply_filter(self):
        """Применение фильтра к загруженному изображению"""
        if self.original_image is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение.")
            return

        kernel_str = self.kernel_entry.get().strip()
        try:
            numbers = list(map(float, kernel_str.split()))
            if len(numbers) != 9:
                raise ValueError("Должно быть ровно 9 чисел.")
            kernel = np.array(numbers, dtype=np.float64).reshape(3, 3)
        except Exception as e:
            messagebox.showerror("Ошибка маски", f"Некорректный ввод маски:\n{str(e)}")
            return

        normalize = self.normalize_var.get()

        self.apply_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.preset_combo.config(state=tk.DISABLED)
        self.progress.grid()
        self.progress.start()

        def task():
            try:
                result = self.manual_convolution(self.original_image, kernel, normalize)
                self.filtered_image = result
                self.root.after(0, self.on_filter_done)
            except Exception as e:
                self.root.after(0, lambda: self.on_filter_error(str(e)))

        threading.Thread(target=task, daemon=True).start()

    def on_filter_done(self):
        """Вызывается после успешного завершения фильтрации"""
        self.progress.stop()
        self.progress.grid_remove()
        self.display_image(self.filtered_image, self.filtered_canvas, "filtered")
        self.save_btn.config(state=tk.NORMAL)
        self.apply_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.preset_combo.config(state="readonly")

    def on_filter_error(self, error_msg):
        """Обработка ошибки при фильтрации"""
        self.progress.stop()
        self.progress.grid_remove()
        messagebox.showerror("Ошибка фильтрации", error_msg)
        self.apply_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.preset_combo.config(state="readonly")

    def save_image(self):
        """Сохранение обработанного изображения"""
        if self.filtered_image is None:
            messagebox.showwarning("Нет результата", "Нечего сохранять.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            pil_img = Image.fromarray(self.filtered_image)
            pil_img.save(file_path)
            messagebox.showinfo("Сохранение", f"Изображение сохранено:\n{file_path}")
        except Exception as e:
            try:
                img_bgr = cv2.cvtColor(self.filtered_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                messagebox.showinfo("Сохранение", f"Изображение сохранено:\n{file_path}")
            except Exception as e2:
                messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить изображение:\n{str(e)}")
