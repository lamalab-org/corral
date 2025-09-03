from scipy.integrate import solve_ivp


def solve_system(ode_func, y0, tspan, params):
    sol = solve_ivp(
        lambda t, y: ode_func(t, y, params), [tspan[0], tspan[-1]], y0, t_eval=tspan
    )
    return sol.t, sol.y.T
