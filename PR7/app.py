import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

class BinarizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Бинаризация изображения (ручная и метод Оцу)")
        self.root.geometry("1200x750")
        self.root.resizable(True, True)

        # Переменные
        self.original_image = None      # Исходное цветное (RGB)
        self.gray_image = None          # Полутоновое (0-255)
        self.binary_image = None        # Бинарное (0 или 255)
        self.histogram = None           # Гистограмма яркостей
        self.current_threshold = 128    # Текущий порог

        # Для отображения в Tkinter
        self.photo_original = None
        self.photo_binary = None

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Верхняя панель управления
        control_frame = tk.Frame(self.root, padx=5, pady=5)
        control_frame.pack(fill=tk.X)

        # Кнопка загрузки
        self.load_btn = tk.Button(control_frame, text="Загрузить изображение",
                                 command=self.load_image, width=18, bg="#2196F3", fg="white")
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(control_frame, text="Файл не выбран", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Разделитель
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # Ручное управление порогом
        manual_frame = tk.LabelFrame(self.root, text="Ручная бинаризация", padx=5, pady=5)
        manual_frame.pack(fill=tk.X, padx=10, pady=5)

        threshold_frame = tk.Frame(manual_frame)
        threshold_frame.pack(fill=tk.X, pady=5)

        tk.Label(threshold_frame, text="Порог t (0-255):").pack(side=tk.LEFT, padx=5)

        self.threshold_var = tk.IntVar(value=128)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0, to=255,
                                         orient=tk.HORIZONTAL, variable=self.threshold_var,
                                         command=self.on_threshold_change)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.threshold_spinbox = tk.Spinbox(threshold_frame, from_=0, to=255,
                                           textvariable=self.threshold_var, width=5,
                                           command=self.on_threshold_spin)
        self.threshold_spinbox.pack(side=tk.LEFT, padx=5)

        self.apply_manual_btn = tk.Button(threshold_frame, text="Применить",
                                         command=self.apply_manual_threshold, width=10)
        self.apply_manual_btn.pack(side=tk.LEFT, padx=5)

        # Автоматическая бинаризация
        auto_frame = tk.LabelFrame(self.root, text="Автоматическая бинаризация (метод Оцу)", padx=5, pady=5)
        auto_frame.pack(fill=tk.X, padx=10, pady=5)

        self.otsu_btn = tk.Button(auto_frame, text="Вычислить порог Оцу и применить",
                                 command=self.apply_otsu, bg="#4CAF50", fg="white", height=2)
        self.otsu_btn.pack(pady=5)

        self.otsu_result_label = tk.Label(auto_frame, text="Оптимальный порог: ", font=("Arial", 10))
        self.otsu_result_label.pack()

        # Область изображений
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Левое изображение (исходное в градациях серого)
        left_frame = tk.LabelFrame(image_frame, text="Исходное (оттенки серого)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.original_canvas = tk.Canvas(left_frame, bg='lightgray')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Правое изображение (бинаризованное)
        right_frame = tk.LabelFrame(image_frame, text="Бинаризованное (0 / 255)")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.binary_canvas = tk.Canvas(right_frame, bg='lightgray')
        self.binary_canvas.pack(fill=tk.BOTH, expand=True)

        # Гистограмма
        hist_frame = tk.LabelFrame(self.root, text="Гистограмма яркости", padx=5, pady=5)
        hist_frame.pack(fill=tk.BOTH, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 2.5))
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=hist_frame)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Кнопка сохранения
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        self.save_btn = tk.Button(bottom_frame, text="Сохранить бинарное изображение",
                                 command=self.save_image, state=tk.DISABLED, bg="#FF9800", fg="white")
        self.save_btn.pack(side=tk.RIGHT)

        # Привязка изменения размера холстов
        self.original_canvas.bind("<Configure>", self.on_resize)
        self.binary_canvas.bind("<Configure>", self.on_resize)

        # Изначально деактивируем элементы, требующие изображения
        self.set_ui_state(False)

    def set_ui_state(self, has_image):
        """Активация/деактивация элементов интерфейса в зависимости от наличия изображения"""
        state = tk.NORMAL if has_image else tk.DISABLED
        self.threshold_slider.config(state=state)
        self.threshold_spinbox.config(state=state)
        self.apply_manual_btn.config(state=state)
        self.otsu_btn.config(state=state)
        self.save_btn.config(state=state if self.binary_image is not None else tk.DISABLED)

    def load_image(self):
        """Загрузка изображения через диалог"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            # Загрузка через PIL (надёжнее с кириллицей)
            pil_img = Image.open(file_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            self.original_image = np.array(pil_img)

            # Преобразование в оттенки серого
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

            # Вычисление гистограммы
            self.compute_histogram()

            # Отображение исходного изображения (в градациях серого)
            self.display_image(self.gray_image, self.original_canvas, "original")

            # Очистка предыдущего бинарного изображения
            self.binary_image = None
            self.binary_canvas.delete("all")
            self.photo_binary = None

            # Обновление имени файла
            self.file_label.config(text=os.path.basename(file_path), fg="black")

            # Активация интерфейса
            self.set_ui_state(True)
            self.save_btn.config(state=tk.DISABLED)

            # Обновление гистограммы (без порога)
            self.update_histogram_plot()

        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить изображение:\n{str(e)}")
            self.original_image = None
            self.gray_image = None
            self.file_label.config(text="Ошибка загрузки", fg="red")
            self.set_ui_state(False)

    def compute_histogram(self):
        """Вычисление гистограммы яркостей (0-255)"""
        if self.gray_image is None:
            return
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        self.histogram = hist.flatten()

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

        # Для бинарного изображения нужен RGB (3 канала) для PIL
        if len(img.shape) == 2:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            display_img = img

        pil_img = Image.fromarray(display_img)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)

        if tag == "original":
            self.photo_original = photo
        else:
            self.photo_binary = photo

        canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)

    def on_resize(self, event):
        """Перерисовка изображений при изменении размера холстов"""
        if self.gray_image is not None:
            self.display_image(self.gray_image, self.original_canvas, "original")
        if self.binary_image is not None:
            self.display_image(self.binary_image, self.binary_canvas, "binary")

    def update_histogram_plot(self, threshold=None):
        """Обновление графика гистограммы. Если threshold указан, рисуется вертикальная линия."""
        if self.histogram is None:
            return

        self.ax.clear()
        x = np.arange(256)
        self.ax.bar(x, self.histogram, width=1.0, color='gray', edgecolor='black', alpha=0.7)
        self.ax.set_xlim([0, 255])
        self.ax.set_xlabel("Яркость")
        self.ax.set_ylabel("Количество пикселей")

        if threshold is not None:
            self.ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f't = {threshold}')
            self.ax.legend()

        self.fig.tight_layout()
        self.canvas_hist.draw()

    def on_threshold_change(self, event=None):
        """Обработка изменения слайдера (только обновление отображения порога на гистограмме)"""
        t = self.threshold_var.get()
        self.update_histogram_plot(threshold=t)

    def on_threshold_spin(self):
        """Обработка изменения спинбокса"""
        t = self.threshold_var.get()
        self.update_histogram_plot(threshold=t)

    def apply_manual_threshold(self):
        """Применение ручного порога и бинаризация"""
        if self.gray_image is None:
            return
        t = self.threshold_var.get()
        self.current_threshold = t
        # Бинаризация: < t -> 0, >= t -> 255
        _, self.binary_image = cv2.threshold(self.gray_image, t, 255, cv2.THRESH_BINARY)
        self.display_image(self.binary_image, self.binary_canvas, "binary")
        self.save_btn.config(state=tk.NORMAL)
        self.update_histogram_plot(threshold=t)

    def apply_otsu(self):
        """Автоматическая бинаризация методом Оцу с полным выводом в консоль"""
        if self.gray_image is None:
            return

        print("\n" + "=" * 80)
        print("                    МЕТОД ОЦУ — АВТОМАТИЧЕСКАЯ БИНАРИЗАЦИЯ")
        print("=" * 80)

        # Ручная реализация Оцу
        hist = self.histogram.astype(np.float64)
        total_pixels = np.sum(hist)

        print(f"\n1. ИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИИ")
        print(f"   Общее количество пикселей N = {int(total_pixels)}")
        print(f"   Диапазон яркостей: 0..255")
        print(f"   Размер изображения: {self.gray_image.shape[1]}x{self.gray_image.shape[0]}")

        if total_pixels == 0:
            print("   ОШИБКА: изображение пустое!")
            return

        # Вероятности pi
        prob = hist / total_pixels

        print(f"\n2. ПОЛНАЯ ГИСТОГРАММА ЯРКОСТЕЙ")
        print(f"   Формат: [Яркость] -> Количество пикселей (Вероятность)")
        print(f"   " + "-" * 50)

        # Выводим все 256 значений гистограммы
        non_zero_count = 0
        for i in range(256):
            if hist[i] > 0:
                print(f"   [{i:3d}] -> {int(hist[i]):8d} пикс. (p = {prob[i]:.6f})")
                non_zero_count += 1

        if non_zero_count == 0:
            print("   ВНИМАНИЕ: Все значения гистограммы равны нулю!")
        else:
            print(f"   " + "-" * 50)
            print(f"   Всего ненулевых уровней яркости: {non_zero_count}")

        # Префиксные суммы вероятностей и математических ожиданий
        q1 = np.cumsum(prob)  # q1(t) = sum_{i=0..t} p_i
        q2 = 1.0 - q1  # q2(t) = 1 - q1(t)

        # Средние значения
        cum_sum = np.cumsum(np.arange(256) * prob)  # sum i * p_i до t
        M1 = np.zeros(256)
        M2 = np.zeros(256)

        with np.errstate(divide='ignore', invalid='ignore'):
            M1 = np.where(q1 > 0, cum_sum / q1, 0)
            M2 = np.where(q2 > 0, (cum_sum[-1] - cum_sum) / q2, 0)

        # Межклассовая дисперсия
        sigma_b_squared = q1 * q2 * (M1 - M2) ** 2

        # Находим t, максимизирующий sigma_b_squared
        valid = (q1 > 0) & (q2 > 0)
        sigma_b_squared[~valid] = -1
        best_t = np.argmax(sigma_b_squared)
        max_sigma = sigma_b_squared[best_t]

        print(f"\n3. ПОЛНЫЙ ПРОЦЕСС ПОИСКА ОПТИМАЛЬНОГО ПОРОГА")
        print(f"   Проверяются все t от 0 до 255")
        print(f"\n   " + "-" * 100)
        print(f"   {'t':>4} | {'q1(t)':>12} | {'q2(t)':>12} | {'M1(t)':>12} | {'M2(t)':>12} | {'σ²_B(t)':>15} | Статус")
        print(f"   " + "-" * 100)

        # Выводим ВСЕ значения t от 0 до 255
        valid_count = 0
        for t in range(256):
            if valid[t]:
                status = "✓"
                valid_count += 1
            else:
                if q1[t] == 0:
                    status = "q1=0"
                elif q2[t] == 0:
                    status = "q2=0"
                else:
                    status = "—"

            sigma_str = f"{sigma_b_squared[t]:15.2f}" if valid[t] else "        —"

            print(f"   {t:4d} | {q1[t]:12.6f} | {q2[t]:12.6f} | {M1[t]:12.4f} | {M2[t]:12.4f} | {sigma_str} | {status}")

        print(f"   " + "-" * 100)
        print(f"   Всего валидных порогов (q1>0 и q2>0): {valid_count} из 256")

        # Находим топ-5 лучших порогов
        sorted_indices = np.argsort(sigma_b_squared)[::-1]  # по убыванию
        print(f"\n4. ТОП-5 ЛУЧШИХ ПОРОГОВ (по убыванию σ²_B):")
        print(f"   " + "-" * 60)
        print(f"   {'Место':>5} | {'t':>4} | {'σ²_B(t)':>15} | {'q1':>12} | {'q2':>12} | {'M1':>10} | {'M2':>10}")
        print(f"   " + "-" * 60)
        for rank, t in enumerate(sorted_indices[:5], 1):
            if valid[t]:
                print(
                    f"   {rank:5d} | {t:4d} | {sigma_b_squared[t]:15.2f} | {q1[t]:12.4f} | {q2[t]:12.4f} | {M1[t]:10.2f} | {M2[t]:10.2f}")

        print(f"\n5. ОПТИМАЛЬНЫЙ ПОРОГ")
        print(f"   " + "=" * 40)
        print(f"   t* = {best_t}")
        print(f"   Максимальная межклассовая дисперсия σ²_B(t*) = {max_sigma:.4f}")
        print(f"\n   Характеристики классов при оптимальном пороге:")
        print(f"   " + "-" * 40)
        print(f"   Класс 1 (фон):     яркости [0 .. {best_t}]")
        print(f"   Класс 2 (объект):  яркости [{best_t + 1} .. 255]")
        print(f"\n   q1({best_t}) = {q1[best_t]:.6f} (доля пикселей в классе 1)")
        print(f"   q2({best_t}) = {q2[best_t]:.6f} (доля пикселей в классе 2)")
        print(f"\n   M1({best_t}) = {M1[best_t]:.4f} (средняя яркость класса 1)")
        print(f"   M2({best_t}) = {M2[best_t]:.4f} (средняя яркость класса 2)")
        print(f"\n   Проверка: q1 + q2 = {q1[best_t] + q2[best_t]:.6f} (должно быть 1.0)")

        self.current_threshold = best_t
        self.threshold_var.set(best_t)

        # Применяем порог
        _, self.binary_image = cv2.threshold(self.gray_image, best_t, 255, cv2.THRESH_BINARY)

        # Подсчёт пикселей после бинаризации
        black_pixels = np.sum(self.binary_image == 0)
        white_pixels = np.sum(self.binary_image == 255)

        print(f"\n6. РЕЗУЛЬТАТ БИНАРИЗАЦИИ")
        print(f"   " + "-" * 40)
        print(f"   Правило преобразования:")
        print(f"   если яркость < {best_t}  → 0 (чёрный)")
        print(f"   если яркость >= {best_t} → 255 (белый)")
        print(f"\n   После бинаризации:")
        print(f"   Чёрных пикселей (0):   {black_pixels:8d} ({100 * black_pixels / total_pixels:.2f}%)")
        print(f"   Белых пикселей (255):  {white_pixels:8d} ({100 * white_pixels / total_pixels:.2f}%)")
        print(f"   Всего пикселей:        {int(total_pixels):8d} (100.00%)")
        print("=" * 80 + "\n")

        self.display_image(self.binary_image, self.binary_canvas, "binary")
        self.save_btn.config(state=tk.NORMAL)
        self.otsu_result_label.config(text=f"Оптимальный порог (Оцу): {best_t}")
        self.update_histogram_plot(threshold=best_t)

    def save_image(self):
        """Сохранение бинарного изображения"""
        if self.binary_image is None:
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить бинарное изображение",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            # PIL сохранение
            pil_img = Image.fromarray(self.binary_image)
            pil_img.save(file_path)
            messagebox.showinfo("Сохранение", f"Изображение сохранено:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))