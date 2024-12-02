# bifurcation.py
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import nonlinsolve


def generate_bifurcation_diagram(f_sym, g_sym, parameters, param_ranges, variables=(sp.Symbol('x'), sp.Symbol('y'))):
    """
    Genera diagramas de bifurcación para múltiples parámetros.

    :param f_sym: Ecuación simbólica para x'
    :param g_sym: Ecuación simbólica para y'
    :param parameters: Lista de símbolos de parámetros a analizar
    :param param_ranges: Diccionario con rangos para cada parámetro, por ejemplo:
                         {param1: (-2, 2), param2: (0, 5)}
    :param variables: Tupla de símbolos de variables (x, y)
    """
    num_params = len(parameters)
    if num_params == 0:
        print("No se han proporcionado parámetros para generar el diagrama de bifurcación.")
        return

    # Determinar filas y columnas para subplots
    if num_params == 1:
        rows, cols = 1, 1
    else:
        cols = 2  # Puedes ajustar el número de columnas según tus preferencias
        rows = (num_params + cols - 1) // cols  # Redondeo hacia arriba

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    # Asegurar que 'axes' sea siempre una lista
    if num_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, param in enumerate(parameters):
        ax = axes[idx]
        equilibria_values = []
        range_min, range_max = param_ranges.get(param, (-2, 2))
        param_values = np.linspace(range_min, range_max, 200)

        # Fijar otros parámetros al valor medio de su rango
        fixed_substitutions = {}
        for other_param in parameters:
            if other_param != param:
                other_min, other_max = param_ranges.get(other_param, (-2, 2))
                fixed_value = (other_min + other_max) / 2
                fixed_substitutions[other_param] = fixed_value

        print(f"\nGenerando diagrama de bifurcación para el parámetro '{param}' con los siguientes valores fijos:")
        for k, v in fixed_substitutions.items():
            print(f"  {k} = {v}")

        for param_val in param_values:
            substitutions = {param: param_val}
            substitutions.update(fixed_substitutions)

            f_sub = f_sym.subs(substitutions)
            g_sub = g_sym.subs(substitutions)

            try:
                # Usar nonlinsolve para sistemas no lineales
                solutions = nonlinsolve([f_sub, g_sub], variables)
                # solutions es un FiniteSet de tuplas
            except Exception as e:
                print(f"Error al resolver para {param} = {param_val}: {e}")
                continue

            # Iterar sobre las soluciones encontradas
            for sol in solutions:
                eq_x = sol[0]
                eq_y = sol[1]

                eq_x_num = eq_x.evalf()
                eq_y_num = eq_y.evalf()

                # Verificar si ambas soluciones son reales
                if eq_x_num.is_real and eq_y_num.is_real:
                    try:
                        equilibria_values.append((param_val, float(eq_x_num), float(eq_y_num)))
                    except (TypeError, ValueError):
                        continue  # Ignorar si no se puede convertir a float
                else:
                    continue  # Ignorar soluciones complejas

        if not equilibria_values:
            ax.text(0.5, 0.5, "No se encontraron puntos de equilibrio.", horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes)
            continue

        # Extraer los valores para graficar
        param_vals = [val[0] for val in equilibria_values]
        x_vals = [val[1] for val in equilibria_values]
        y_vals = [val[2] for val in equilibria_values]

        # Graficar los puntos de equilibrio
        ax.plot(param_vals, x_vals, 'b.', label='x equilibria')
        ax.plot(param_vals, y_vals, 'r.', label='y equilibria')

        ax.set_xlabel(f'Valor del parámetro {param}')
        ax.set_ylabel('Puntos de equilibrio')
        ax.set_title(f'Diagrama de Bifurcación para {param}')
        ax.legend()
        ax.grid(True)

    # Ocultar subplots no utilizados si los hay
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
