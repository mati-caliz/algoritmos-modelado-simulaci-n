# bifurcation.py
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import solve


def generate_bifurcation_diagram(f_sym, g_sym, parameter, param_range, ax, variables=(sp.Symbol('x'), sp.Symbol('y'))):
    equilibria_values = []
    param_values = np.linspace(param_range[0], param_range[1], 200)

    for param_val in param_values:
        substitutions = {parameter: param_val}
        f_sub = f_sym.subs(substitutions)
        g_sub = g_sym.subs(substitutions)
        solutions = solve([f_sub, g_sub], variables, dict=True)

        for sol in solutions:
            try:
                eq_x = sol.get(variables[0], variables[0])
                eq_y = sol.get(variables[1], variables[1])
                eq_x_num = float(eq_x.evalf())
                eq_y_num = float(eq_y.evalf())
                equilibria_values.append((param_val, eq_x_num, eq_y_num))
            except (TypeError, ValueError):
                continue

    if not equilibria_values:
        ax.text(0.5, 0.5, f'No se encontraron puntos de equilibrio para {parameter}',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return

    param_vals = [val[0] for val in equilibria_values]
    x_vals = [val[1] for val in equilibria_values]
    y_vals = [val[2] for val in equilibria_values]

    # Dibujar las bifurcaciones en el eje proporcionado
    ax.plot(param_vals, x_vals, 'b.', label='x equilibria')
    ax.plot(param_vals, y_vals, 'r.', label='y equilibria')
    ax.set_xlabel(f'Valor del parámetro {parameter}')
    ax.set_ylabel('Puntos de equilibrio')
    ax.set_title(f'Diagrama de Bifurcación para {parameter}')
    ax.legend()
    ax.grid(True)
    plt.figure(figsize=(10, 6))
    plt.plot(param_vals, x_vals, 'b.', label='x equilibria')
    plt.plot(param_vals, y_vals, 'r.', label='y equilibria')
    plt.xlabel(f'Valor del parámetro {parameter}')
    plt.ylabel('Puntos de equilibrio')
    plt.title('Diagrama de Bifurcación')
    plt.legend()
    plt.grid(True)
    plt.show()
