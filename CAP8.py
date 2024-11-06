import streamlit as st
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon



st.set_page_config(page_title="Introducción a la Optimización", layout="wide")
st.title("Programación Lineal Entera")
st.write("### Una guía interactiva en español para comprender la Programación Lineal Entera")

st.header("Introducción a la Programación Lineal Entera (PLE)")
st.write("""
La Programación Lineal Entera (PLE) es una técnica de optimización en la que se busca maximizar o minimizar una 
función objetivo sujeta a restricciones lineales, donde las variables deben tomar valores enteros. A continuación, veremos ejemplos,
explicaciones y resolveremos algunos ejercicios sobre métodos de PLE.
""")

st.header("Ejemplo: Anna’s Cozy Home Furnishings")
st.write("""
Anna's Cozy Home Furnishings (ACHF) fabrica dos tipos de mesas de madera: 
- **Farmhouse** (modelo simple)
- **Designer** (modelo elegante).

Condiciones:
- Tiempo de ensamblaje: 3 horas para Farmhouse, 5 horas para Designer.
- Disponibilidad semanal de ensamblaje: 71 horas.
- Corte externo mínimo requerido: 30 horas por semana.

Definimos:
- **x** = número de mesas Farmhouse producidas
- **y** = número de mesas Designer producidas

Formulación del Problema:
""")
st.latex(r'''
\text{Maximizar } P = 100x + 250y
''')
st.latex(r'''
\text{sujeto a:}
\begin{cases}
3x + 5y \leq 71 \\
x + 2y \geq 30 \\
x, y \geq 0 \text{ y enteros}
\end{cases}
''')

st.subheader("Visualización de la Región Factible")

x_vals = np.linspace(0, 25, 400)
y1 = (71 - 3 * x_vals) / 5
y2 = (30 - x_vals) / 2

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y1, label=r'$3x + 5y \leq 71$')
plt.plot(x_vals, y2, label=r'$x + 2y \geq 30$')

y3 = np.maximum(0, np.minimum(y1, y2))
plt.fill_between(x_vals, 0, y3, where=(y3 >= 0), color='gray', alpha=0.3)

plt.xlim((0, 25))
plt.ylim((0, 25))
plt.xlabel('Número de mesas Farmhouse (x)')
plt.ylabel('Número de mesas Designer (y)')
plt.legend()
plt.title("Región Factible para el Problema de Anna")

st.pyplot(plt)

st.subheader("Resolución del Problema usando Programación Lineal Continua")
c = [-100, -250]
A = [[3, 5], [-1, -2]]
b = [71, -30]
res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")

if res.success:
    st.write("**Solución óptima fraccional (sin restricciones enteras):**")
    st.write(f"Farmhouse (x) = {res.x[0]:.2f}, Designer (y) = {res.x[1]:.2f}")
    st.write(f"Ganancia máxima aproximada: ${-res.fun:.2f}")
else:
    st.write("No se encontró una solución factible.")

st.write("""
Dado que la solución fraccional no es realista, utilizaremos el método de Branch and Bound para encontrar una solución entera.
""")

def branch_and_bound():
    bounds = [(0, None), (0, None)]
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    if not result.success:
        return None, None
    
    stack = [(result.x[0], result.x[1], result.fun)]
    best_sol = None
    best_val = float('inf')
    
    while stack:
        x, y, val = stack.pop()
        if x.is_integer() and y.is_integer():
            if val < best_val:
                best_sol = (int(x), int(y))
                best_val = val
        else:
            if not x.is_integer():
                stack.append((np.floor(x), y, val))
                stack.append((np.ceil(x), y, val))
            if not y.is_integer():
                stack.append((x, np.floor(y), val))
                stack.append((x, np.ceil(y), val))
    
    return best_sol, -best_val

sol, max_profit = branch_and_bound()
if sol:
    st.write(f"**Solución entera óptima usando Branch and Bound:**")
    st.write(f"Farmhouse (x) = {sol[0]}, Designer (y) = {sol[1]}")
    st.write(f"Ganancia máxima: ${max_profit}")
else:
    st.write("No se encontró una solución entera factible.")

st.header("Ejercicios")
st.write("""
A continuación, puedes probar resolver problemas adicionales de Programación Lineal Entera. 
Ingresa tus propios valores y selecciona un método de resolución.
""")

st.subheader("Definir un nuevo problema de PLE")
num_vars = st.number_input("Número de variables:", min_value=2, max_value=5, value=3)
objective = st.text_input("Función objetivo (ej. '4x1 + 3x2 + 3x3'):")
constraints = st.text_area("Restricciones (una por línea, ej. '4x1 + 2x2 + x3 <= 10'):")

if st.button("Resolver problema"):
    st.write("Resolviendo el problema...")

    st.write("Función de objetivo y restricciones procesadas (en construcción).")

st.header("Ejercicios del Documento")
st.write("""
Ejercicio 8.1: Resuelve el siguiente problema usando el Método Branch and Bound:
- Maximizar P(x1, x2, x3) = 4x1 + 3x2 + 3x3
- Sujeto a:
  - 4x1 + 2x2 + x3 ≤ 10
  - 3x1 + 4x2 + 2x3 ≤ 14
  - 2x1 + x2 + 3x3 ≤ 7
  - x1, x2, x3 enteros y ≥ 0.
""")
