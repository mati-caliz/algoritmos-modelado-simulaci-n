import sympy as sp
from dynamic_system import DynamicSystem

def main():
    x, y, a, b = sp.symbols('x y a b')
    f_x = a-x**2+b+x
    g_x = -y+b

    system = DynamicSystem(
        f_sym=f_x,
        g_sym=g_x,
        parameters={a,b} # None if no parameters
    )

    system.run_full_analysis()


if __name__ == "__main__":
    main()
