import numpy as np
import casadi as ca

from dynamics import (
    NX,
    NU,
    N_SAT,
    NX_ONE,
    NU_ONE,
    make_casadi_functions,
    extract_dynamics_params,
    propagate_open_loop_np,
)


def get_state_slices(i):
    """
    Возвращает индексы состояния i-го аппарата в полном векторе X.

    Порядок состояния одного аппарата:
        [x, vx, y, vy, z, vz, psi, theta]
    """

    start = i * NX_ONE

    return {
        "x": start + 0,
        "vx": start + 1,
        "y": start + 2,
        "vy": start + 3,
        "z": start + 4,
        "vz": start + 5,
        "psi": start + 6,
        "theta": start + 7,
    }


def get_control_slices(i):
    """
    Возвращает индексы управления i-го аппарата в полном векторе U.

    Порядок управления одного аппарата:
        [f, omega_psi, omega_theta]
    """

    start = i * NU_ONE

    return {
        "f": start + 0,
        "omega_psi": start + 1,
        "omega_theta": start + 2,
    }


def get_position_vector(X, k, i):
    """
    Возвращает вектор положения rho_i(t_k) = [x_i, y_i, z_i]^T.
    """

    idx = get_state_slices(i)

    return ca.vertcat(
        X[idx["x"], k],
        X[idx["y"], k],
        X[idx["z"], k],
    )


def get_velocity_vector(X, k, i):
    """
    Возвращает вектор относительной скорости v_i(t_k) = [vx_i, vy_i, vz_i]^T.
    """

    idx = get_state_slices(i)

    return ca.vertcat(
        X[idx["vx"], k],
        X[idx["vy"], k],
        X[idx["vz"], k],
    )


def build_initial_guess(config):
    """
    Строит начальное приближение для NLP.

    В качестве первого приближения используется свободное движение при нулевом
    управлении. Такое приближение согласовано с динамикой и хорошо подходит
    для начального запуска прямого метода.
    """

    X0 = np.array(config["initial_state"]["X0"], dtype=float)
    time_grid = np.array(config["time"]["time_grid"], dtype=float)

    params = extract_dynamics_params(config)

    N = len(time_grid) - 1
    U_guess = np.zeros((N, NU))

    X_guess = propagate_open_loop_np(
        X0=X0,
        U_grid=U_guess,
        time_grid=time_grid,
        params=params,
    )

    return X_guess, U_guess


def add_control_constraints(opti, U, config):
    """
    Добавляет ограничения на модуль тяги и скорости изменения углов.
    """

    N = config["time"]["N"]

    f_max = config["control"]["f_max_m_s2"]
    omega_psi_max = config["control"]["omega_psi_max_rad_s"]
    omega_theta_max = config["control"]["omega_theta_max_rad_s"]

    for k in range(N):
        for i in range(N_SAT):
            idx_u = get_control_slices(i)

            f = U[idx_u["f"], k]
            omega_psi = U[idx_u["omega_psi"], k]
            omega_theta = U[idx_u["omega_theta"], k]

            opti.subject_to(f >= 0.0)
            opti.subject_to(f <= f_max)

            opti.subject_to(omega_psi >= -omega_psi_max)
            opti.subject_to(omega_psi <= omega_psi_max)

            opti.subject_to(omega_theta >= -omega_theta_max)
            opti.subject_to(omega_theta <= omega_theta_max)


def add_angle_constraints(opti, X, config):
    """
    Добавляет ограничения на углы ориентации тяги.

    Эти ограничения задаются достаточно естественно:
        psi   -- азимутальный угол в плоскости xOy,
        theta -- угол отклонения от плоскости xOy.
    """

    N = config["time"]["N"]

    psi_min = config["control"]["psi_min"]
    psi_max = config["control"]["psi_max"]

    theta_min = config["control"]["theta_min"]
    theta_max = config["control"]["theta_max"]

    for k in range(N + 1):
        for i in range(N_SAT):
            idx_x = get_state_slices(i)

            psi = X[idx_x["psi"], k]
            theta = X[idx_x["theta"], k]

            opti.subject_to(psi >= psi_min)
            opti.subject_to(psi <= psi_max)

            opti.subject_to(theta >= theta_min)
            opti.subject_to(theta <= theta_max)


def add_dynamics_constraints(opti, X, U, rk4_fun, config):
    """
    Добавляет ограничения множественной стрельбы:
        X_{k+1} = F_dt(X_k, U_k, t_k).
    """

    N = config["time"]["N"]
    time_grid = np.array(config["time"]["time_grid"], dtype=float)

    for k in range(N):
        X_next = rk4_fun(X[:, k], U[:, k], float(time_grid[k]))
        opti.subject_to(X[:, k + 1] == X_next)


def build_objective(opti, X, U, config):
    """
    Формирует функционал качества.

    На подготовительном участке [0, t_obs) используется вес w_prep.
    На участке наблюдения [t_obs, T] используется вес w_obs.

    Основной штраф:
        - ошибка положения относительно требуемых вершин;
        - ошибка скорости относительно нулевой относительной скорости.

    Дополнительно добавлены малые регуляризирующие штрафы:
        - за использование тяги;
        - за скорости изменения углов.
    """

    N = config["time"]["N"]
    dt = config["time"]["dt_s"]
    t_obs = config["time"]["t_obs_s"]
    time_grid = np.array(config["time"]["time_grid"], dtype=float)

    targets = np.array(config["formation"]["targets_m"], dtype=float)

    rho_tol = config["formation"]["rho_tol_m"]
    v_tol = config["cost"]["v_tol_m_s"]

    w_prep = config["cost"]["w_prep"]
    w_obs = config["cost"]["w_obs"]

    eps_thrust = config["cost"]["eps_thrust"]
    eps_angle_rate = config["cost"]["eps_angle_rate"]

    f_max = config["control"]["f_max_m_s2"]
    omega_psi_max = config["control"]["omega_psi_max_rad_s"]
    omega_theta_max = config["control"]["omega_theta_max_rad_s"]

    J = 0.0

    for k in range(N):
        t_k = time_grid[k]
        w_k = w_obs if t_k >= t_obs else w_prep

        for i in range(N_SAT):
            rho_i = get_position_vector(X, k, i)
            v_i = get_velocity_vector(X, k, i)

            rho_star_i = ca.DM(targets[i, :])

            e_rho = rho_i - rho_star_i
            e_v = v_i

            J += (
                w_k
                * (
                    ca.sumsqr(e_rho) / rho_tol**2
                    + ca.sumsqr(e_v) / v_tol**2
                )
                * dt
            )

    for k in range(N):
        for i in range(N_SAT):
            idx_u = get_control_slices(i)

            f = U[idx_u["f"], k]
            omega_psi = U[idx_u["omega_psi"], k]
            omega_theta = U[idx_u["omega_theta"], k]

            J += eps_thrust * (f / f_max) ** 2 * dt

            J += (
                eps_angle_rate
                * (
                    (omega_psi / omega_psi_max) ** 2
                    + (omega_theta / omega_theta_max) ** 2
                )
                * dt
            )

    return J


def build_ocp(config):
    """
    Строит задачу оптимального управления прямым методом.

    Возвращает:
        opti -- объект CasADi Opti,
        X    -- матрица переменных состояния размерности (NX, N + 1),
        U    -- матрица переменных управления размерности (NU, N),
        J    -- функционал качества.
    """

    N = config["time"]["N"]

    opti = ca.Opti()

    X = opti.variable(NX, N + 1)
    U = opti.variable(NU, N)

    _, rk4_fun = make_casadi_functions(config)

    X0 = np.array(config["initial_state"]["X0"], dtype=float)

    opti.subject_to(X[:, 0] == X0)

    add_dynamics_constraints(opti, X, U, rk4_fun, config)
    add_control_constraints(opti, U, config)
    add_angle_constraints(opti, X, config)

    J = build_objective(opti, X, U, config)
    opti.minimize(J)

    X_guess, U_guess = build_initial_guess(config)

    opti.set_initial(X, X_guess.T)
    opti.set_initial(U, U_guess.T)

    solver_options = {
        "expand": False,
        "print_time": True,
    }

    ipopt_options = {
        "max_iter": config["solver"]["ipopt_max_iter"],
        "tol": config["solver"]["ipopt_tol"],
        "acceptable_tol": config["solver"]["ipopt_acceptable_tol"],
        "print_level": config["solver"]["print_level"],
        "mu_strategy": "adaptive",
    }

    opti.solver("ipopt", solver_options, ipopt_options)

    return opti, X, U, J


def solve_ocp(config):
    """
    Строит и решает задачу оптимального управления.

    Если IPOPT не находит оптимальное решение, функция всё равно пытается вернуть
    последнее доступное приближение через opti.debug.value(...). Это удобно для
    диагностики неудачных запусков.
    """

    opti, X, U, J = build_ocp(config)

    try:
        sol = opti.solve()

        status = "success"

        X_sol = np.array(sol.value(X)).T
        U_sol = np.array(sol.value(U)).T
        J_sol = float(sol.value(J))

    except RuntimeError as error:
        print()
        print("IPOPT не завершил решение успешно.")
        print("Будет возвращено последнее доступное приближение.")
        print()
        print(error)

        status = "failed"

        X_sol = np.array(opti.debug.value(X)).T
        U_sol = np.array(opti.debug.value(U)).T
        J_sol = float(opti.debug.value(J))

    solution = {
        "status": status,
        "objective": J_sol,
        "X": X_sol,
        "U": U_sol,
        "time_grid": np.array(config["time"]["time_grid"], dtype=float),
        "control_time_grid": np.array(config["time"]["time_grid"][:-1], dtype=float),
    }

    return solution


def print_ocp_size(config):
    """
    Печатает размерность построенной NLP-задачи.
    """

    N = config["time"]["N"]

    n_state_vars = NX * (N + 1)
    n_control_vars = NU * N
    n_total_vars = n_state_vars + n_control_vars

    n_dynamic_constraints = NX * N

    print("Размерность задачи оптимального управления:")
    print(f"N = {N}")
    print(f"NX = {NX}")
    print(f"NU = {NU}")
    print()
    print(f"Переменные состояния:   {n_state_vars}")
    print(f"Переменные управления: {n_control_vars}")
    print(f"Всего переменных:      {n_total_vars}")
    print()
    print(f"Ограничения динамики:  {n_dynamic_constraints}")


if __name__ == "__main__":
    from config import build_config

    config = build_config()

    print_ocp_size(config)

    print()
    print("Построение OCP...")

    opti, X, U, J = build_ocp(config)

    print("OCP успешно построена.")