
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sympy import symbols, solve, diff, integrate, simplify
from sympy.parsing.sympy_parser import parse_expr
import math

class MathApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Math Toolkit")
        self.geometry("1000x700")
        self.configure(bg='#f0f0f0')
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.create_calculator_tab()
        self.create_equation_solver_tab()
        self.create_graph_plotter_tab()
        self.create_calculus_tab()
        self.create_matrix_tab()
        self.create_statistics_tab()
        self.create_help_tab()
        
    def create_calculator_tab(self):
        calc_frame = ttk.Frame(self.notebook)
        self.notebook.add(calc_frame, text="Calculator")
        self.calc_display = tk.Entry(calc_frame, font=('Arial', 20), justify='right', bd=10)
        self.calc_display.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky='ew')
        buttons = [
            ['7','8','9','/','sin'],
            ['4','5','6','*','cos'],
            ['1','2','3','-','tan'],
            ['0','.','=','+','sqrt'],
            ['(',')','^ ','%','log'],
            ['C','CE','π','e','ln'],
            ['x!','abs','exp','//','**']
        ]
        for i, row in enumerate(buttons, start=1):
            for j, btn in enumerate(row):
                cmd = lambda x=btn: self.calculator_click(x)
                button = tk.Button(calc_frame, text=btn, font=('Arial', 14), command=cmd, width=8, height=2, bg='#4CAF50' if btn == '=' else '#e0e0e0')
                button.grid(row=i, column=j, padx=5, pady=5)
        calc_frame.grid_columnconfigure(tuple(range(5)), weight=1)
    
    def calculator_click(self, key):
        current = self.calc_display.get()
        if key == '=':
            try:
                expr = current.replace('^', '**').replace('π', str(math.pi)).replace('e', str(math.e))
                expr = expr.replace('sqrt', 'math.sqrt').replace('sin', 'math.sin').replace('cos', 'math.cos').replace('tan', 'math.tan').replace('log', 'math.log10').replace('ln', 'math.log')
                expr = expr.replace('abs', 'abs').replace('exp', 'math.exp')
                if 'x!' in expr:
                    result = math.factorial(int(float(expr.replace('x!', ''))))
                else:
                    result = eval(expr)
                self.calc_display.delete(0, tk.END)
                self.calc_display.insert(0, str(result))
            except: 
                messagebox.showerror("Error", "Invalid")
        elif key == 'C': 
            self.calc_display.delete(0, tk.END)
        elif key == 'CE': 
            self.calc_display.delete(0, tk.END)
            self.calc_display.insert(0, current[:-1])
        elif key in ['sin','cos','tan','sqrt','log','ln','abs','exp']: 
            self.calc_display.insert(tk.END, f'{key}(')
        else: 
            self.calc_display.insert(tk.END, key)
    
    def create_equation_solver_tab(self):
        solver_frame = ttk.Frame(self.notebook)
        self.notebook.add(solver_frame, text="Equation Solver")
        tk.Label(solver_frame, text="Enter equation (e.g., x**2 - 4 = 0):", font=('Arial', 12)).pack(pady=10)
        self.equation_entry = tk.Entry(solver_frame, font=('Arial', 14), width=50)
        self.equation_entry.pack(pady=10)
        tk.Label(solver_frame, text="Variable:", font=('Arial', 12)).pack(pady=5)
        self.var_entry = tk.Entry(solver_frame, font=('Arial', 14), width=10)
        self.var_entry.insert(0, 'x')
        self.var_entry.pack(pady=5)
        tk.Button(solver_frame, text="Solve", font=('Arial', 12), command=self.solve_equation, bg='#2196F3', fg='white', width=20, height=2).pack(pady=20)
        self.solution_text = scrolledtext.ScrolledText(solver_frame, height=15, font=('Arial', 11))
        self.solution_text.pack(padx=20, pady=10, fill='both', expand=True)
    
    def solve_equation(self):
        try:
            equation = self.equation_entry.get()
            var_name = self.var_entry.get()
            if '=' in equation:
                left, right = equation.split('=')
                expr = parse_expr(left) - parse_expr(right)
            else:
                expr = parse_expr(equation)
            var = symbols(var_name)
            solutions = solve(expr, var)
            self.solution_text.delete(1.0, tk.END)
            self.solution_text.insert(tk.END, f"Equation: {equation}\n\nSolutions:\n\n")
            if solutions:
                for i, sol in enumerate(solutions, 1):
                    self.solution_text.insert(tk.END, f"{var_name}_{i} = {sol}\n")
            else:
                self.solution_text.insert(tk.END, "No solutions found.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_graph_plotter_tab(self):
        graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(graph_frame, text="Graph Plotter")
        control_frame = tk.Frame(graph_frame)
        control_frame.pack(side='top', fill='x', padx=10, pady=10)
        tk.Label(control_frame, text="Function:", font=('Arial', 11)).pack(side='left', padx=5)
        self.func_entry = tk.Entry(control_frame, font=('Arial', 11), width=30)
        self.func_entry.insert(0, 'x**2')
        self.func_entry.pack(side='left', padx=5)
        tk.Label(control_frame, text="X min:", font=('Arial', 11)).pack(side='left', padx=5)
        self.xmin_entry = tk.Entry(control_frame, font=('Arial', 11), width=8)
        self.xmin_entry.insert(0, '-10')
        self.xmin_entry.pack(side='left', padx=2)
        tk.Label(control_frame, text="X max:", font=('Arial', 11)).pack(side='left', padx=5)
        self.xmax_entry = tk.Entry(control_frame, font=('Arial', 11), width=8)
        self.xmax_entry.insert(0, '10')
        self.xmax_entry.pack(side='left', padx=2)
        tk.Button(control_frame, text="Plot", command=self.plot_graph, bg='#4CAF50', fg='white', font=('Arial', 11)).pack(side='left', padx=10)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
    
    def plot_graph(self):
        try:
            func_str = self.func_entry.get()
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            x = np.linspace(xmin, xmax, 500)
            x_sym = symbols('x')
            expr = parse_expr(func_str)
            f = lambda val: float(expr.subs(x_sym, val))
            y = [f(val) for val in x]
            self.ax.clear()
            self.ax.plot(x, y, 'b-', linewidth=2)
            self.ax.grid(True, alpha=0.3)
            self.ax.axhline(y=0, color='k', linewidth=0.5)
            self.ax.axvline(x=0, color='k', linewidth=0.5)
            self.ax.set_xlabel('x', fontsize=12)
            self.ax.set_ylabel('f(x)', fontsize=12)
            self.ax.set_title(f'f(x) = {func_str}', fontsize=14)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_calculus_tab(self):
        calculus_frame = ttk.Frame(self.notebook)
        self.notebook.add(calculus_frame, text="Calculus")
        tk.Label(calculus_frame, text="Enter function:", font=('Arial', 12)).pack(pady=10)
        self.calculus_entry = tk.Entry(calculus_frame, font=('Arial', 14), width=50)
        self.calculus_entry.insert(0, 'x**3 + 2*x**2 - 5*x + 3')
        self.calculus_entry.pack(pady=10)
        button_frame = tk.Frame(calculus_frame)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Differentiate", command=self.differentiate, bg='#FF9800', fg='white', font=('Arial', 11), width=15, height=2).pack(side='left', padx=10)
        tk.Button(button_frame, text="Integrate", command=self.integrate_func, bg='#9C27B0', fg='white', font=('Arial', 11), width=15, height=2).pack(side='left', padx=10)
        tk.Button(button_frame, text="Simplify", command=self.simplify_func, bg='#00BCD4', fg='white', font=('Arial', 11), width=15, height=2).pack(side='left', padx=10)
        self.calculus_result = scrolledtext.ScrolledText(calculus_frame, height=12, font=('Arial', 11))
        self.calculus_result.pack(padx=20, pady=10, fill='both', expand=True)
    
    def differentiate(self):
        try:
            x = symbols('x')
            expr = parse_expr(self.calculus_entry.get())
            derivative = diff(expr, x)
            self.calculus_result.delete(1.0, tk.END)
            self.calculus_result.insert(tk.END, f"f(x) = {self.calculus_entry.get()}\n\nf'(x) = {derivative}\n\nf''(x) = {diff(derivative, x)}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def integrate_func(self):
        try:
            x = symbols('x')
            expr = parse_expr(self.calculus_entry.get())
            integral = integrate(expr, x)
            self.calculus_result.delete(1.0, tk.END)
            self.calculus_result.insert(tk.END, f"f(x) = {self.calculus_entry.get()}\n\n∫f(x)dx = {integral} + C\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def simplify_func(self):
        try:
            x = symbols('x')
            expr = parse_expr(self.calculus_entry.get())
            simplified = simplify(expr)
            self.calculus_result.delete(1.0, tk.END)
            self.calculus_result.insert(tk.END, f"Original: {self.calculus_entry.get()}\n\nSimplified: {simplified}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_matrix_tab(self):
        matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(matrix_frame, text="Matrix Operations")
        tk.Label(matrix_frame, text="Matrix A (rows separated by ;)", font=('Arial', 11)).pack(pady=5)
        self.matrix_a = tk.Entry(matrix_frame, font=('Arial', 11), width=50)
        self.matrix_a.insert(0, '1,2,3; 4,5,6; 7,8,9')
        self.matrix_a.pack(pady=5)
        tk.Label(matrix_frame, text="Matrix B", font=('Arial', 11)).pack(pady=5)
        self.matrix_b = tk.Entry(matrix_frame, font=('Arial', 11), width=50)
        self.matrix_b.insert(0, '9,8,7; 6,5,4; 3,2,1')
        self.matrix_b.pack(pady=5)
        button_frame = tk.Frame(matrix_frame)
        button_frame.pack(pady=15)
        operations = [("Add", self.matrix_add), ("Multiply", self.matrix_multiply), ("Det", self.matrix_det), ("Inverse", self.matrix_inverse), ("Transpose", self.matrix_transpose)]
        for text, cmd in operations:
            tk.Button(button_frame, text=text, command=cmd, width=12, font=('Arial', 10), bg='#607D8B', fg='white').pack(side='left', padx=5)
        self.matrix_result = scrolledtext.ScrolledText(matrix_frame, height=15, font=('Courier', 10))
        self.matrix_result.pack(padx=20, pady=10, fill='both', expand=True)
    
    def parse_matrix(self, s):
        return np.array([[float(x.strip()) for x in row.split(',')] for row in s.split(';')])
    
    def matrix_add(self):
        try:
            A = self.parse_matrix(self.matrix_a.get())
            B = self.parse_matrix(self.matrix_b.get())
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"A + B:\n\n{A + B}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def matrix_multiply(self):
        try:
            A = self.parse_matrix(self.matrix_a.get())
            B = self.parse_matrix(self.matrix_b.get())
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"A × B:\n\n{np.matmul(A, B)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def matrix_det(self):
        try:
            A = self.parse_matrix(self.matrix_a.get())
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Determinant:\n\n{np.linalg.det(A)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def matrix_inverse(self):
        try:
            A = self.parse_matrix(self.matrix_a.get())
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Inverse:\n\n{np.linalg.inv(A)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def matrix_transpose(self):
        try:
            A = self.parse_matrix(self.matrix_a.get())
            self.matrix_result.delete(1.0, tk.END)
            self.matrix_result.insert(tk.END, f"Transpose:\n\n{A.T}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_statistics_tab(self):
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        tk.Label(stats_frame, text="Enter data (comma-separated):", font=('Arial', 12)).pack(pady=10)
        self.stats_entry = tk.Entry(stats_frame, font=('Arial', 14), width=60)
        self.stats_entry.insert(0, '12, 15, 18, 20, 22, 25, 28, 30, 35, 40')
        self.stats_entry.pack(pady=10)
        tk.Button(stats_frame, text="Calculate", command=self.calculate_stats, bg='#E91E63', fg='white', font=('Arial', 12), width=20, height=2).pack(pady=20)
        self.stats_result = scrolledtext.ScrolledText(stats_frame, height=18, font=('Arial', 11))
        self.stats_result.pack(padx=20, pady=10, fill='both', expand=True)
    
    def calculate_stats(self):
        try:
            data = np.array([float(x.strip()) for x in self.stats_entry.get().split(',')])
            self.stats_result.delete(1.0, tk.END)
            self.stats_result.insert(tk.END, f"Data: {data}\n\nCount: {len(data)}\n\nMean: {np.mean(data):.4f}\nMedian: {np.median(data):.4f}\nStd Dev: {np.std(data):.4f}\nVariance: {np.var(data):.4f}\n\nMin: {np.min(data):.4f}\nMax: {np.max(data):.4f}\nRange: {np.max(data) - np.min(data):.4f}\n\nQ1: {np.percentile(data, 25):.4f}\nQ3: {np.percentile(data, 75):.4f}\nIQR: {np.percentile(data, 75) - np.percentile(data, 25):.4f}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_help_tab(self):
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="📖 User Manual")
        help_text = scrolledtext.ScrolledText(help_frame, height=30, font=('Arial', 11), wrap=tk.WORD)
        help_text.pack(padx=20, pady=20, fill='both', expand=True)
        manual = """MATH TOOLKIT USER MANUAL

CALCULATOR TAB:
Basic operations: +, -, *, /, % (modulus), ^ (power)
Functions: sin(x), cos(x), tan(x), sqrt(x), log(x), ln(x), abs(x), exp(x), x!
Constants: π (pi), e (Euler's number)
Example: sin(30) + log(100)

EQUATION SOLVER:
Enter equations like: x**2 - 4 = 0
Supports: Linear, quadratic, cubic equations
Variable: Default is 'x', but you can change it
Example: 2*x + 5 = 11 gives x = 3

GRAPH PLOTTER:
Enter function: x**2, sin(x), log(x), etc.
Set X range: Adjust min/max values
For log functions: Use positive X values (e.g., X min = 0.1)
Example: Plot x**3 from -5 to 5

CALCULUS TOOLS:
Differentiate: Find f'(x) and f''(x)
Integrate: Find ∫f(x)dx
Simplify: Reduce expressions
Example: Differentiate x**2 gives 2*x

MATRIX OPERATIONS:
Format: rows separated by ; and elements by ,
Example: 1,2,3; 4,5,6; 7,8,9
Operations: Add, Multiply, Determinant, Inverse, Transpose

STATISTICS:
Enter data: Comma-separated numbers
Example: 10, 20, 30, 40, 50
Calculates: Mean, median, std dev, variance, quartiles

TIPS:
- Use parentheses for clarity: (2+3)*4
- Trig functions use radians
- For degrees: multiply by π/180
- Modulus example: 10 % 3 = 1

Created by Parth Atomar
"""
        help_text.insert('1.0', manual)
        help_text.config(state='disabled')

if __name__ == "__main__":
    app = MathApplication()
    app.mainloop()

