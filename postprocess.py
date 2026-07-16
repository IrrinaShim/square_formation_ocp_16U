import json
from pathlib import Path
import numpy as np

from config import ProjectConfig
from dynamics import N_SAT, NX_ONE, NU_ONE


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_solution(results_dir):
    """
    Загружает config.json и solution.npz из папки результатов.
    """

    results_dir = Path(results_dir)

    config_path = results_dir / "config.json"
    solution_path = results_dir / "solution.npz"

    if not config_path.exists():
        raise FileNotFoundError(f"Не найден файл {config_path}")

    if not solution_path.exists():
        raise FileNotFoundError(f"Не найден файл {solution_path}")

    config = load_json(config_path)
    data = np.load(solution_path, allow_pickle=True)

    solution = {
        "X": data["X"],
        "U": data["U"],
        "time_grid": data["time_grid"],
        "control_time_grid": data["control_time_grid"],
        "objective": float(data["objective"][0]),
        "status": str(data["status"][0]),
        "targets_m": data["targets_m"],
    }

    return config, solution


def get_propulsion_parameters(config):
    """
    Извлекает параметры двигательной установки из секции control
    и проверяет их согласованность.
    """

    control = config["control"]

    required_keys = [
        "mass_kg",
        "F_max_N",
        "f_max_m_s2",
        "specific_impulse_s",
        "total_impulse_Ns",
        "g0_m_s2",
    ]

    missing_keys = [key for key in required_keys if key not in control]

    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(
            "В секции config['control'] отсутствуют параметры: "
            f"{missing}"
        )

    params = {
        key: float(control[key])
        for key in required_keys
    }

    for name, value in params.items():
        if value <= 0.0:
            raise ValueError(f"Параметр {name} должен быть положительным.")

    expected_f_max = params["F_max_N"] / params["mass_kg"]

    if not np.isclose(
        params["f_max_m_s2"],
        expected_f_max,
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise ValueError(
            "Несогласованные параметры управления: "
            f"f_max_m_s2={params['f_max_m_s2']:.12e}, однако "
            f"F_max_N / mass_kg={expected_f_max:.12e}."
        )

    return params


def compute_propulsion_feasibility(config):
    """
    Оценивает необходимые ускорение, тягу, импульс и расход топлива
    для удержания аппарата при z = d на рабочем интервале.
    """

    propulsion = get_propulsion_parameters(config)

    q = float(config["orbit"]["q"])
    d_m = float(config["formation"]["d_m"])

    T_s = float(config["time"]["T_s"])
    t_obs_s = float(config["time"]["t_obs_s"])
    duration_s = T_s - t_obs_s

    required_acceleration_m_s2 = q**2 * d_m
    required_force_N = propulsion["mass_kg"] * required_acceleration_m_s2

    hold_delta_v_m_s = required_acceleration_m_s2 * duration_s
    hold_impulse_Ns = required_force_N * duration_s
    hold_propellant_kg = hold_impulse_Ns / (
        propulsion["specific_impulse_s"] * propulsion["g0_m_s2"]
    )
    hold_impulse_fraction = hold_impulse_Ns / propulsion["total_impulse_Ns"]

    return {
        "required_acceleration_m_s2": float(required_acceleration_m_s2),
        "required_force_N": float(required_force_N),
        "acceleration_margin": float(
            propulsion["f_max_m_s2"] / required_acceleration_m_s2
        ),
        "thrust_margin": float(propulsion["F_max_N"] / required_force_N),
        "observation_duration_s": float(duration_s),
        "hold_delta_v_m_s": float(hold_delta_v_m_s),
        "hold_impulse_Ns": float(hold_impulse_Ns),
        "hold_propellant_kg": float(hold_propellant_kg),
        "hold_impulse_fraction": float(hold_impulse_fraction),
        "hold_impulse_percent": float(100.0 * hold_impulse_fraction),
        "is_thrust_sufficient": bool(
            propulsion["F_max_N"] >= required_force_N
        ),
    }


def get_sat_position(X_grid, sat_index):
    """
    Возвращает положение ведомого аппарата:
        rho_i = [x_i, y_i, z_i].

    sat_index = 0, 1, 2 соответствует ведомым аппаратам 1, 2, 3.
    """

    start = sat_index * NX_ONE

    x = X_grid[:, start + 0]
    y = X_grid[:, start + 2]
    z = X_grid[:, start + 4]

    return np.column_stack([x, y, z])


def get_all_vertices(X_grid):
    """
    Возвращает вершины текущей конфигурации.

    Вершина 0 -- опорный аппарат:
        rho_0 = [0, 0, 0].

    Вершины 1, 2, 3 -- три ведомых аппарата.
    """

    n_points = X_grid.shape[0]

    rho0 = np.zeros((n_points, 3))
    rho1 = get_sat_position(X_grid, 0)
    rho2 = get_sat_position(X_grid, 1)
    rho3 = get_sat_position(X_grid, 2)

    return rho0, rho1, rho2, rho3


def distance(a, b):
    """
    Евклидово расстояние между двумя массивами точек.
    """

    return np.linalg.norm(a - b, axis=1)


def compute_shape_metrics(solution, config):
    """
    Считает стороны и диагонали текущей конфигурации.

    Для идеального квадрата:
        L01 = L02 = L13 = L23 = d,
        D03 = D12 = sqrt(2) d.
    """

    X_grid = solution["X"]
    time_grid = solution["time_grid"]

    d = float(config["formation"]["d_m"])
    diag_ref = np.sqrt(2.0) * d

    rho0, rho1, rho2, rho3 = get_all_vertices(X_grid)

    L01 = distance(rho0, rho1)
    L02 = distance(rho0, rho2)
    L13 = distance(rho1, rho3)
    L23 = distance(rho2, rho3)

    D03 = distance(rho0, rho3)
    D12 = distance(rho1, rho2)

    shape = {
        "t": time_grid,
        "L01": L01,
        "L02": L02,
        "L13": L13,
        "L23": L23,
        "D03": D03,
        "D12": D12,
        "err_L01": L01 - d,
        "err_L02": L02 - d,
        "err_L13": L13 - d,
        "err_L23": L23 - d,
        "err_D03": D03 - diag_ref,
        "err_D12": D12 - diag_ref,
        "rel_err_L01": (L01 - d) / d,
        "rel_err_L02": (L02 - d) / d,
        "rel_err_L13": (L13 - d) / d,
        "rel_err_L23": (L23 - d) / d,
        "rel_err_D03": (D03 - diag_ref) / diag_ref,
        "rel_err_D12": (D12 - diag_ref) / diag_ref,
    }

    return shape


def compute_position_errors(solution, config):
    """
    Считает ошибки положения ведомых аппаратов относительно требуемых вершин.
    """

    X_grid = solution["X"]
    time_grid = solution["time_grid"]

    targets = np.array(config["formation"]["targets_m"], dtype=float)

    result = {
        "t": time_grid,
    }

    for i in range(N_SAT):
        rho_i = get_sat_position(X_grid, i)
        e_i = rho_i - targets[i]

        result[f"ex{i + 1}"] = e_i[:, 0]
        result[f"ey{i + 1}"] = e_i[:, 1]
        result[f"ez{i + 1}"] = e_i[:, 2]
        result[f"e_norm{i + 1}"] = np.linalg.norm(e_i, axis=1)

    return result


def compute_delta_v(solution, config):
    """
    Считает накопленную характеристическую скорость для каждого ведомого аппарата:
        Delta v_i(t) = integral_0^t f_i(tau) dtau.

    Управление задано на интервалах, поэтому массив строится на узлах time_grid.
    """

    U_grid = solution["U"]
    time_grid = solution["time_grid"]

    N = U_grid.shape[0]

    delta_v = {
        "t": time_grid,
    }

    for i in range(N_SAT):
        start = i * NU_ONE
        f = U_grid[:, start + 0]

        dv_i = np.zeros(N + 1)

        for k in range(N):
            dt = time_grid[k + 1] - time_grid[k]
            dv_i[k + 1] = dv_i[k] + f[k] * dt

        delta_v[f"delta_v{i + 1}"] = dv_i

    return delta_v


def compute_propulsion_history(solution, delta_v, config):
    """
    Переводит накопленную характеристическую скорость в физический импульс
    и оценку израсходованной массы топлива:

        I_i(t) = m Delta v_i(t),
        Delta m_i(t) = I_i(t) / (I_sp g_0).

    Все массивы заданы на узлах time_grid.
    """

    propulsion = get_propulsion_parameters(config)

    mass_kg = propulsion["mass_kg"]
    specific_impulse_s = propulsion["specific_impulse_s"]
    g0_m_s2 = propulsion["g0_m_s2"]
    total_impulse_Ns = propulsion["total_impulse_Ns"]

    history = {
        "t": np.asarray(solution["time_grid"], dtype=float),
    }

    for i in range(N_SAT):
        number = i + 1
        dv_i = np.asarray(delta_v[f"delta_v{number}"], dtype=float)

        impulse_i = mass_kg * dv_i
        propellant_i = impulse_i / (specific_impulse_s * g0_m_s2)
        impulse_fraction_i = impulse_i / total_impulse_Ns

        history[f"delta_v{number}"] = dv_i
        history[f"impulse{number}_Ns"] = impulse_i
        history[f"propellant{number}_kg"] = propellant_i
        history[f"impulse_fraction{number}"] = impulse_fraction_i

    return history


def compute_force_history(solution, config):
    """
    Считает физическую силу тяги F_i = m f_i на интервалах управления.
    """

    propulsion = get_propulsion_parameters(config)

    U_grid = np.asarray(solution["U"], dtype=float)
    control_time_grid = np.asarray(solution["control_time_grid"], dtype=float)

    history = {
        "t": control_time_grid,
    }

    for i in range(N_SAT):
        number = i + 1
        start = i * NU_ONE

        f_i = U_grid[:, start]
        force_i = propulsion["mass_kg"] * f_i

        history[f"f{number}_m_s2"] = f_i
        history[f"F{number}_N"] = force_i

    return history


def compute_observation_summary(
    shape,
    position_errors,
    propulsion_history,
    force_history,
    config,
):
    """
    Считает основные численные характеристики на участке наблюдения.
    """

    t = shape["t"]

    t_obs = float(config["time"]["t_obs_s"])
    d = float(config["formation"]["d_m"])
    rho_tol = float(config["formation"]["rho_tol_m"])
    diag_ref = np.sqrt(2.0) * d

    propulsion = get_propulsion_parameters(config)
    feasibility = compute_propulsion_feasibility(config)

    obs_mask = t >= t_obs
    control_obs_mask = force_history["t"] >= t_obs

    k_obs = int(np.argmin(np.abs(t - t_obs)))

    side_names = ["L01", "L02", "L13", "L23"]
    diag_names = ["D03", "D12"]

    summary = {
        "observation_interval": {
            "t_obs_s": t_obs,
            "T_s": float(config["time"]["T_s"]),
            "duration_s": float(config["time"]["T_s"] - t_obs),
        },
        "reference_values": {
            "side_m": d,
            "diagonal_m": float(diag_ref),
            "position_tolerance_m": rho_tol,
        },
        "position_errors": {},
        "shape_errors": {},
        "propulsion_parameters": propulsion,
        "propulsion_feasibility": feasibility,
        "delta_v": {},
        "propulsion": {},
    }

    all_position_errors = []

    for i in range(N_SAT):
        e_name = f"e_norm{i + 1}"
        e_obs = position_errors[e_name][obs_mask]

        all_position_errors.append(e_obs)

        summary["position_errors"][f"sat_{i + 1}"] = {
            "max_m": float(np.max(e_obs)),
            "rms_m": float(np.sqrt(np.mean(e_obs**2))),
            "final_m": float(e_obs[-1]),
        }

    all_position_errors = np.concatenate(all_position_errors)

    summary["position_errors"]["all_satellites"] = {
        "max_m": float(np.max(all_position_errors)),
        "rms_m": float(np.sqrt(np.mean(all_position_errors**2))),
        "max_to_tolerance_ratio": float(np.max(all_position_errors) / rho_tol),
    }

    for name in side_names:
        err = shape[name][obs_mask] - d

        summary["shape_errors"][name] = {
            "max_abs_m": float(np.max(np.abs(err))),
            "rms_m": float(np.sqrt(np.mean(err**2))),
            "max_abs_relative": float(np.max(np.abs(err)) / d),
        }

    for name in diag_names:
        err = shape[name][obs_mask] - diag_ref

        summary["shape_errors"][name] = {
            "max_abs_m": float(np.max(np.abs(err))),
            "rms_m": float(np.sqrt(np.mean(err**2))),
            "max_abs_relative": float(np.max(np.abs(err)) / diag_ref),
        }

    for i in range(N_SAT):
        number = i + 1

        dv = propulsion_history[f"delta_v{number}"]
        impulse = propulsion_history[f"impulse{number}_Ns"]
        propellant = propulsion_history[f"propellant{number}_kg"]
        impulse_fraction = propulsion_history[f"impulse_fraction{number}"]

        force = force_history[f"F{number}_N"]
        f = force_history[f"f{number}_m_s2"]

        summary["delta_v"][f"sat_{number}"] = {
            "final_m_s": float(dv[-1]),
            "observation_m_s": float(dv[-1] - dv[k_obs]),
        }

        summary["propulsion"][f"sat_{number}"] = {
            "max_acceleration_m_s2": float(np.max(f)),
            "max_force_N": float(np.max(force)),
            "max_force_obs_N": float(np.max(force[control_obs_mask])),
            "total_impulse_Ns": float(impulse[-1]),
            "observation_impulse_Ns": float(
                impulse[-1] - impulse[k_obs]
            ),
            "propellant_consumption_kg": float(propellant[-1]),
            "observation_propellant_kg": float(
                propellant[-1] - propellant[k_obs]
            ),
            "total_impulse_fraction": float(impulse_fraction[-1]),
            "total_impulse_percent": float(
                100.0 * impulse_fraction[-1]
            ),
        }

    return summary


def save_table_csv(table, path, column_order=None):
    """
    Сохраняет словарь массивов одинаковой длины в CSV.
    """

    if column_order is None:
        column_order = list(table.keys())

    data = np.column_stack([table[key] for key in column_order])

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(column_order),
        comments="",
    )


def save_shape_metrics(shape, results_dir):
    path = results_dir / "shape_metrics.csv"

    columns = [
        "t",
        "L01",
        "L02",
        "L13",
        "L23",
        "D03",
        "D12",
        "err_L01",
        "err_L02",
        "err_L13",
        "err_L23",
        "err_D03",
        "err_D12",
        "rel_err_L01",
        "rel_err_L02",
        "rel_err_L13",
        "rel_err_L23",
        "rel_err_D03",
        "rel_err_D12",
    ]

    save_table_csv(shape, path, columns)

    return path


def save_position_errors(position_errors, results_dir):
    path = results_dir / "position_errors_postprocessed.csv"

    columns = [
        "t",
        "ex1",
        "ey1",
        "ez1",
        "e_norm1",
        "ex2",
        "ey2",
        "ez2",
        "e_norm2",
        "ex3",
        "ey3",
        "ez3",
        "e_norm3",
    ]

    save_table_csv(position_errors, path, columns)

    return path


def save_delta_v(delta_v, results_dir):
    path = results_dir / "delta_v.csv"

    columns = [
        "t",
        "delta_v1",
        "delta_v2",
        "delta_v3",
    ]

    save_table_csv(delta_v, path, columns)

    return path


def save_propulsion_history(propulsion_history, results_dir):
    path = results_dir / "propulsion_history.csv"

    columns = ["t"]

    for i in range(N_SAT):
        number = i + 1
        columns.extend(
            [
                f"delta_v{number}",
                f"impulse{number}_Ns",
                f"propellant{number}_kg",
                f"impulse_fraction{number}",
            ]
        )

    save_table_csv(propulsion_history, path, columns)

    return path


def save_force_history(force_history, results_dir):
    path = results_dir / "force_history.csv"

    columns = ["t"]

    for i in range(N_SAT):
        number = i + 1
        columns.extend(
            [
                f"f{number}_m_s2",
                f"F{number}_N",
            ]
        )

    save_table_csv(force_history, path, columns)

    return path


def print_observation_summary(summary):
    print()
    print("Постобработка завершена.")
    print()
    print("Участок наблюдения:")
    print(
        f"t = [{summary['observation_interval']['t_obs_s']:.1f}, "
        f"{summary['observation_interval']['T_s']:.1f}] с"
    )

    print()
    print("Ошибки положения:")
    all_sat = summary["position_errors"]["all_satellites"]
    print(f"max ||e|| = {all_sat['max_m']:.6f} м")
    print(f"RMS ||e|| = {all_sat['rms_m']:.6f} м")
    print(f"max ||e|| / rho_tol = {all_sat['max_to_tolerance_ratio']:.6f}")

    print()
    print("Ошибки сторон квадрата:")
    for name in ["L01", "L02", "L13", "L23"]:
        item = summary["shape_errors"][name]
        print(
            f"{name}: max |err| = {item['max_abs_m']:.6f} м, "
            f"max rel = {item['max_abs_relative']:.6e}"
        )

    print()
    print("Ошибки диагоналей:")
    for name in ["D03", "D12"]:
        item = summary["shape_errors"][name]
        print(
            f"{name}: max |err| = {item['max_abs_m']:.6f} м, "
            f"max rel = {item['max_abs_relative']:.6e}"
        )

    print()
    print("Итоговая характеристическая скорость:")
    for i in range(N_SAT):
        item = summary["delta_v"][f"sat_{i + 1}"]
        print(
            f"sat {i + 1}: Delta v = {item['final_m_s']:.6f} м/с, "
            f"на участке наблюдения = {item['observation_m_s']:.6f} м/с"
        )

    print()
    print("Проверка двигательной установки:")

    feasibility = summary["propulsion_feasibility"]

    print(
        "Требуемое ускорение для удержания z=d: "
        f"{feasibility['required_acceleration_m_s2']:.8e} м/с^2"
    )
    print(
        "Требуемая сила тяги для удержания z=d: "
        f"{feasibility['required_force_N']:.6f} Н"
    )
    print(f"Запас по тяге: {feasibility['thrust_margin']:.6f}")
    print(
        "Оценочный импульс удержания: "
        f"{feasibility['hold_impulse_Ns']:.6f} Н·с"
    )
    print(
        "Оценочный расход топлива: "
        f"{feasibility['hold_propellant_kg']:.6f} кг"
    )
    print(
        "Доля полного импульса: "
        f"{feasibility['hold_impulse_percent']:.3f} %"
    )
    print(
        "Тяги достаточно: "
        f"{'да' if feasibility['is_thrust_sufficient'] else 'нет'}"
    )

    print()
    print("Фактические затраты двигательной установки:")

    for i in range(N_SAT):
        item = summary["propulsion"][f"sat_{i + 1}"]

        print(f"sat {i + 1}:")
        print(f"  max F = {item['max_force_N']:.6f} Н")
        print(f"  импульс = {item['total_impulse_Ns']:.6f} Н·с")
        print(
            "  расход топлива = "
            f"{item['propellant_consumption_kg']:.6f} кг"
        )
        print(
            "  доля полного импульса = "
            f"{item['total_impulse_percent']:.3f} %"
        )


def run_postprocess(results_dir):
    results_dir = Path(results_dir)

    config, solution = load_solution(results_dir)

    shape = compute_shape_metrics(solution, config)
    position_errors = compute_position_errors(solution, config)
    delta_v = compute_delta_v(solution, config)
    propulsion_history = compute_propulsion_history(
        solution=solution,
        delta_v=delta_v,
        config=config,
    )
    force_history = compute_force_history(solution, config)

    summary = compute_observation_summary(
        shape=shape,
        position_errors=position_errors,
        propulsion_history=propulsion_history,
        force_history=force_history,
        config=config,
    )

    shape_path = save_shape_metrics(shape, results_dir)
    position_errors_path = save_position_errors(position_errors, results_dir)
    delta_v_path = save_delta_v(delta_v, results_dir)
    propulsion_history_path = save_propulsion_history(
        propulsion_history,
        results_dir,
    )
    force_history_path = save_force_history(force_history, results_dir)

    summary_path = results_dir / "postprocess_summary.json"
    save_json(summary, summary_path)

    print_observation_summary(summary)

    print()
    print("Сохранённые файлы постобработки:")
    print(shape_path)
    print(position_errors_path)
    print(delta_v_path)
    print(propulsion_history_path)
    print(force_history_path)
    print(summary_path)


def main():
    results_dir = ProjectConfig().results_dir
    run_postprocess(results_dir)


if __name__ == "__main__":
    main()

