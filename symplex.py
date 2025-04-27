import tkinter as tk
from tkinter import messagebox
from fractions import Fraction
import copy
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, LongTable, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class SimplexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Симплекс-метод")
        self.history = []
        self.selected_pivot = None
        self.ratio_labels = []
        self.table_entries = []
        self.base_var_entries = []
        self.header_labels = []
        self.initial_matrix = None
        self.initial_base_vars = None
        self.initial_free_vars = None
        self.setup_initial_input()

    def setup_initial_input(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Количество переменных (k):", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=5)
        self.k_entry = tk.Entry(self.root, font=("Arial", 12))
        self.k_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Количество ограничений (m):", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=5)
        self.m_entry = tk.Entry(self.root, font=("Arial", 12))
        self.m_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Button(self.root, text="Далее", font=("Arial", 12), command=self.create_input_table, bg="#4CAF50", fg="white").grid(row=2, column=0, columnspan=2, pady=10)

    def create_input_table(self):
        try:
            self.k = int(self.k_entry.get())
            self.m = int(self.m_entry.get())
            if self.k <= 0 or self.m <= 0:
                raise ValueError("Введите положительные числа!")
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные положительные числа для k и m!")
            return

        for widget in self.root.winfo_children():
            widget.destroy()

        self.entries = []
        self.base_vars = []
        self.free_vars_entries = []
        self.free_vars = ["Св.ч"] + [f"x{i+1}" for i in range(self.k)]

        tk.Label(self.root, text="Св.ч", font=("Arial", 12, "bold"), bg="#D3D3D3", relief="ridge").grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
        for j in range(self.k):
            entry = tk.Entry(self.root, width=8, font=("Arial", 12))
            entry.grid(row=0, column=j+2, padx=2, pady=2)
            entry.insert(0, f"x{j+1}")
            self.free_vars_entries.append(entry)

        for i in range(self.m):
            row_entries = []
            base_var_entry = tk.Entry(self.root, width=5, font=("Arial", 12))
            base_var_entry.grid(row=i+1, column=0, padx=2, pady=2)
            base_var_entry.insert(0, f"x{i+self.k+1}")
            self.base_vars.append(base_var_entry)

            for j in range(self.k + 1):
                entry = tk.Entry(self.root, width=8, font=("Arial", 12))
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.entries.append(row_entries)

        tk.Label(self.root, text="F", font=("Arial", 12, "bold"), bg="#D3D3D3", relief="ridge").grid(row=self.m+1, column=0, padx=2, pady=2, sticky="nsew")
        self.obj_entries = []
        for j in range(self.k + 1):
            entry = tk.Entry(self.root, width=8, font=("Arial", 12))
            entry.grid(row=self.m+1, column=j+1, padx=2, pady=2)
            entry.insert(0, "0")
            self.obj_entries.append(entry)

        tk.Button(self.root, text="Отобразить таблицу", font=("Arial", 12), command=self.display_table, bg="#4CAF50", fg="white").grid(row=self.m+2, column=0, columnspan=self.k+3, pady=10)

    def parse_fraction(self, text):
        try:
            if '/' in text:
                num, denom = map(int, text.split('/'))
                return Fraction(num, denom)
            return Fraction(int(text))
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Некорректный ввод: {text}")

    def format_number(self, fraction):
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"

    def show_ratios(self, j):
        for label in self.ratio_labels:
            label.config(text="")
        if j > 0:
            try:
                for i in range(self.m):
                    free_term = self.parse_fraction(self.table_entries[i][0].get())
                    element = self.parse_fraction(self.table_entries[i][j].get())
                    if element != 0:
                        ratio = free_term / element
                        if ratio > 0:
                            self.ratio_labels[i].config(text=f"{float(ratio):.2f}")
            except ValueError as e:
                messagebox.showerror("Ошибка", str(e))
                self.clear_ratios()

    def clear_ratios(self, event=None):
        for label in self.ratio_labels:
            label.config(text="")

    def select_pivot(self, i, j):
        if i < self.m:
            try:
                element = self.parse_fraction(self.table_entries[i][j].get())
                if element != 0:
                    if self.selected_pivot:
                        prev_i, prev_j = self.selected_pivot
                        self.table_entries[prev_i][prev_j].config(bg="#F5F5DC")
                    
                    self.table_entries[i][j].config(bg="#FFFF99")
                    self.selected_pivot = (i, j)
                else:
                    messagebox.showerror("Ошибка", "Пивотный элемент не может быть нулем!")
            except ValueError as e:
                messagebox.showerror("Ошибка", str(e))
        else:
            messagebox.showerror("Ошибка", "Выберите элемент из строк ограничений!")

    def update_matrix_from_entries(self):
        try:
            self.matrix = []
            self.base_vars_list = []
            for i in range(self.m):
                row = [self.parse_fraction(entry.get()) for entry in self.table_entries[i]]
                self.matrix.append(row)
                base_var = self.base_var_entries[i].get().strip()
                if not base_var:
                    raise ValueError("Все базисные переменные должны быть именованы!")
                self.base_vars_list.append(base_var)
            
            obj_row = [self.parse_fraction(entry.get()) for entry in self.table_entries[self.m]]
            self.matrix.append(obj_row)
            self.base_vars_list.append("F")
        except ValueError as e:
            raise ValueError(str(e))

    def display_table(self):
        try:
            self.free_vars = ["Св.ч"] + [entry.get().strip() for entry in self.free_vars_entries]
            if not all(self.free_vars):
                raise ValueError("Все свободные переменные должны быть именованы!")

            self.matrix = []
            self.base_vars_list = []
            for i in range(self.m):
                row = [self.parse_fraction(entry.get()) for entry in self.entries[i]]
                self.matrix.append(row)
                base_var = self.base_vars[i].get().strip()
                if not base_var:
                    raise ValueError("Все базисные переменные должны быть именованы!")
                self.base_vars_list.append(base_var)
            
            self.obj_row = [self.parse_fraction(entry.get()) for entry in self.obj_entries]
            self.matrix.append(self.obj_row)
            self.base_vars_list.append("F")
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        if not self.initial_matrix:
            self.initial_matrix = copy.deepcopy(self.matrix)
            self.initial_base_vars = self.base_vars_list.copy()
            self.initial_free_vars = self.free_vars.copy()

        self.history.append((copy.deepcopy(self.matrix), self.base_vars_list.copy(), self.free_vars.copy()))
        self.selected_pivot = None

        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_table_widgets()
        self.update_table_values()
        self.add_buttons()

    def create_table_widgets(self):
        """Создает виджеты таблицы один раз"""
        self.table_frame = tk.Frame(self.root)
        self.table_frame.grid(row=0, column=0, columnspan=self.k+4)

        # Заголовки
        self.header_labels = []
        for j, header in enumerate(self.free_vars):
            label = tk.Label(self.table_frame, text=header, font=("Arial", 12, "bold"), 
                           bg="#D3D3D3", relief="ridge", width=10)
            label.grid(row=0, column=j+1, padx=2, pady=2, sticky="nsew")
            self.header_labels.append(label)
        
        # Базовые переменные
        tk.Label(self.table_frame, text="БП", font=("Arial", 12, "bold"), 
                bg="#D3D3D3", relief="ridge", width=8).grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        
        # Отношения
        tk.Label(self.table_frame, text="Св.ч/эл.", font=("Arial", 12, "bold"), 
                bg="#D3D3D3", relief="ridge", width=10).grid(row=0, column=self.k+2, padx=2, pady=2, sticky="nsew")

        # Создаем все виджеты ячеек
        self.base_var_entries = []
        self.table_entries = []
        self.ratio_labels = []
        
        for i in range(self.m + 1):
            # Базовые переменные
            base_var = tk.Entry(self.table_frame, width=8, font=("Arial", 12), 
                              bg="#F5F5DC", relief="ridge")
            base_var.grid(row=i+1, column=0, padx=2, pady=2, sticky="nsew")
            if i == self.m:
                base_var.config(state="readonly")
            self.base_var_entries.append(base_var)
            
            # Ячейки таблицы
            row_entries = []
            for j in range(self.k + 1):
                entry = tk.Entry(self.table_frame, width=10, font=("Arial", 12), 
                                bg="#F5F5DC", relief="ridge")
                entry.grid(row=i+1, column=j+1, padx=2, pady=2, sticky="nsew")
                if i < self.m:
                    entry.bind("<Button-1>", lambda e, row=i, col=j: self.select_pivot(row, col))
                    if j > 0:
                        entry.bind("<Enter>", lambda e, col=j: self.show_ratios(col))
                        entry.bind("<Leave>", self.clear_ratios)
                row_entries.append(entry)
            self.table_entries.append(row_entries)
            
            # Отношения
            ratio_label = tk.Label(self.table_frame, text="", font=("Arial", 12), 
                                 bg="#F5F5DC", relief="ridge", width=10)
            ratio_label.grid(row=i+1, column=self.k+2, padx=2, pady=2, sticky="nsew")
            self.ratio_labels.append(ratio_label)

    def update_table_values(self):
        """Обновляет значения в таблице без пересоздания"""
        for j, header in enumerate(self.free_vars):
            self.header_labels[j].config(text=header)
        
        for i in range(self.m + 1):
            self.base_var_entries[i].config(state="normal")
            self.base_var_entries[i].delete(0, tk.END)
            self.base_var_entries[i].insert(0, self.base_vars_list[i])
            if i == self.m:
                self.base_var_entries[i].config(state="readonly")
            
            for j in range(self.k + 1):
                self.table_entries[i][j].delete(0, tk.END)
                self.table_entries[i][j].insert(0, self.format_number(self.matrix[i][j]))
                self.table_entries[i][j].config(bg="#F5F5DC")

    def add_buttons(self):
        """Добавляет кнопки управления"""
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=1, column=0, columnspan=self.k+4, pady=10)
        
        tk.Label(self.root, text="Кликните на элемент таблицы для выбора пивота", 
                font=("Arial", 12)).grid(row=2, column=0, columnspan=self.k+4, pady=5)
        
        buttons = [
            ("Выполнить итерацию", "#4CAF50", self.perform_iteration),
            ("Вернуться назад", "#FF5733", self.undo),
            ("Завершить", "#007BFF", self.finish),
            ("Добавить строку", "#FFA500", self.add_row)
        ]
        
        for i, (text, color, command) in enumerate(buttons):
            tk.Button(self.button_frame, text=text, font=("Arial", 12), 
                    command=command, bg=color, fg="white").grid(row=0, column=i, padx=5)

    def add_row(self):
        try:
            self.update_matrix_from_entries()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        self.m += 1
        new_row = [Fraction(0) for _ in range(self.k + 1)]
        self.matrix.insert(self.m - 1, new_row)
        new_base_var = f"x{self.k + self.m}"
        self.base_vars_list.insert(self.m - 1, new_base_var)
        self.history.append((copy.deepcopy(self.matrix), self.base_vars_list.copy(), self.free_vars.copy()))
        
        # При добавлении строки пересоздаем таблицу полностью
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_table_widgets()
        self.update_table_values()
        self.add_buttons()

    def perform_iteration(self):
        if not self.selected_pivot:
            messagebox.showerror("Ошибка", "Сначала выберите пивотный элемент!")
            return

        try:
            self.update_matrix_from_entries()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        pivot_row, pivot_col = self.selected_pivot
        pivot = self.matrix[pivot_row][pivot_col]

        new_matrix = copy.deepcopy(self.matrix)

        for i in range(self.m + 1):
            if i != pivot_row:
                for j in range(self.k + 1):
                    if j != pivot_col:
                        new_matrix[i][j] = self.matrix[i][j] - (self.matrix[i][pivot_col] * self.matrix[pivot_row][j]) / pivot

        for j in range(self.k + 1):
            if j != pivot_col:
                new_matrix[pivot_row][j] = self.matrix[pivot_row][j] / pivot

        for i in range(self.m + 1):
            if i != pivot_row:
                new_matrix[i][pivot_col] = self.matrix[i][pivot_col] / (-pivot)

        new_matrix[pivot_row][pivot_col] = Fraction(1) / pivot

        self.matrix = new_matrix

        old_base_var = self.base_vars_list[pivot_row]
        old_free_var = self.free_vars[pivot_col]
        self.base_vars_list[pivot_row] = old_free_var
        self.free_vars[pivot_col] = old_base_var

        self.history.append((copy.deepcopy(self.matrix), self.base_vars_list.copy(), self.free_vars.copy()))
        self.selected_pivot = None
        self.update_table_values()
        self.clear_ratios()

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            prev_state = self.history[-1]
            self.matrix = copy.deepcopy(prev_state[0])
            self.base_vars_list = prev_state[1].copy()
            self.free_vars = prev_state[2].copy()
            self.m = len(self.matrix) - 1
            self.selected_pivot = None
            
            # При отмене изменений пересоздаем таблицу если изменилось количество строк
            if len(self.table_entries) != len(self.matrix):
                for widget in self.root.winfo_children():
                    widget.destroy()
                self.create_table_widgets()
                self.update_table_values()
                self.add_buttons()
            else:
                self.update_table_values()
                self.clear_ratios()
        else:
            response = messagebox.askyesno("Начало истории", "Вы достигли начальной таблицы. Хотите вернуться к редактированию исходной таблицы?")
            if response:
                self.history = []
                self.selected_pivot = None
                self.ratio_labels = []
                self.table_entries = []
                self.base_var_entries = []
                self.header_labels = []
                self.create_input_table_with_data()

    def finish(self):
        try:
            self.update_matrix_from_entries()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return

        pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

        pdf_file = "simplex_history.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        title_style = styles["Heading1"]
        subtitle_style = styles["Italic"]
        title_style.fontName = 'DejaVuSans'
        subtitle_style.fontName = 'DejaVuSans'
        title_style.fontSize = 18
        title_style.textColor = colors.darkblue
        subtitle_style.fontSize = 12
        subtitle_style.textColor = colors.darkgreen

        elements.append(Paragraph("История симплекс-таблицы", title_style))
        elements.append(Spacer(1, 20))

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A4A4A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FDF5E6')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ])

        for idx, (matrix, base_vars, free_vars) in enumerate(self.history):
            if idx == 0:
                title = "Начальная таблица"
            else:
                title = f"Итерация {idx}"
            
            elements.append(Paragraph(title, subtitle_style))
            elements.append(Spacer(1, 10))

            table_data = [["БП"] + free_vars]
            for i in range(len(matrix)):
                row = [base_vars[i]] + [self.format_number(val) for val in matrix[i]]
                table_data.append(row)

            table = LongTable(table_data, colWidths=[40] + [50] * len(free_vars), repeatRows=1)
            table.setStyle(table_style)
            elements.append(table)
            elements.append(Spacer(1, 20))

        doc.build(elements)
        messagebox.showinfo("Успех", f"PDF-документ создан: {pdf_file}")
        self.root.quit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimplexApp(root)
    app.run()