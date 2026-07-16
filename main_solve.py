import argparse
import json
from pathlib import Path
import numpy as np

from config import build_config, save_config, make_result_dirs, ProjectConfig
from ocp_builder import solve_ocp
from dynamics import N_SAT, NX_ONE, NU_ONE


def update_config_from_args(config, args):
    """
    Позволяет менять некоторые параметры запуска через командную строку,
    не редактируя config.py вручную.
    """

    if args.N is not None:
        config["time"]["N"] = int(args.N)
        config["time"]["dt_s"] = config["time"]["T_s"] / config["time"]["N"]
        config["time"]["time_grid"] = np.linspace(
            0.0,
            config["time"]["T_s"],
            config["time"]["N"] + 1,
        ).tolist()

    if args.max_iter is not None:
        config["solver"]["ipopt_max_iter"] = int(args.max_iter)

    if args.print_level is not None:
        config["solver"]["print_level"] = int(args.print_level)

    if args.w_prep is not None:
        config["cost"]["w_prep"] = float(args.w_prep)

    if args.eps_thrust is not None:
        config["cost"]["eps_thrust"] = float(args.eps_thrust)

    if args.eps_angle_rate is not None:
        config["cost"]["eps_angle_rate"] = float(args.eps_angle_rate)

    return config


def get_propulsion_parameters(config):
    """
    Извлекает физические параметры двигательной установки и проверяет
    согласованность ограничения на ускорение с массой и максимальной тягой.

    В секции config["control"] должны присутствовать поля:
        mass_kg,
        F_max_N,
        f_max_m_s2,
        specific_impulse_s,
        total_impulse_Ns,
        g0_m_s2.
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

    mass_kg = float(control["mass_kg"])
    F_max_N = float(control["F_max_N"])
    f_max_m_s2 = float(control["f_max_m_s2"])
    specific_impulse_s = float(control["specific_impulse_s"])
    total_impulse_Ns = float(control["total_impulse_Ns"])
    g0_m_s2 = float(control["g0_m_s2"])

    positive_values = {
        "mass_kg": mass_kg,
        "F_max_N": F_max_N,
        "f_max_m_s2": f_max_m_s2,
        "specific_impulse_s": specific_impulse_s,
        "total_impulse_Ns": total_impulse_Ns,
        "g0_m_s2": g0_m_s2,
    }

    for name, value in positive_values.items():
        if value <= 0.0:
            raise ValueError(
                f"Параметр {name} должен быть положительным."
            )

    expected_f_max = F_max_N / mass_kg

    if not np.isclose(
        f_max_m_s2,
        expected_f_max,
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise ValueError(
            "Несогласованные параметры управления: "
            f"f_max_m_s2={f_max_m_s2:.12e}, однако "
            f"F_max_N / mass_kg={expected_f_max:.12e}."
        )

    return {
        "mass_kg": mass_kg,
        "F_max_N": F_max_N,
        "f_max_m_s2": f_max_m_s2,
        "specific_impulse_s": specific_impulse_s,
        "total_impulse_Ns": total_impulse_Ns,
        "g0_m_s2": g0_m_s2,
    }


def compute_propulsion_feasibility(config):
    """
    Оценивает возможность статического удержания аппаратов с z = d.

    Для уравнения
        z_ddot = -q^2 z + a_z
    требуемое ускорение при z = d и z_dot = 0 равно q^2 d.
    """

    propulsion = get_propulsion_parameters(config)

    q = float(config["orbit"]["q"])
    d_m = float(config["formation"]["d_m"])

    T_s = float(config["time"]["T_s"])
    t_obs_s = float(config["time"]["t_obs_s"])
    observation_duration_s = T_s - t_obs_s

    required_acceleration_m_s2 = q**2 * d_m
    required_force_N = (
        propulsion["mass_kg"] * required_acceleration_m_s2
    )

    acceleration_margin = (
        propulsion["f_max_m_s2"] / required_acceleration_m_s2
    )
    thrust_margin = propulsion["F_max_N"] / required_force_N

    hold_delta_v_m_s = (
        required_acceleration_m_s2 * observation_duration_s
    )
    hold_impulse_Ns = required_force_N * observation_duration_s
    hold_propellant_kg = hold_impulse_Ns / (
        propulsion["specific_impulse_s"] * propulsion["g0_m_s2"]
    )
    hold_impulse_fraction = (
        hold_impulse_Ns / propulsion["total_impulse_Ns"]
    )

    return {
        "required_acceleration_m_s2": float(
            required_acceleration_m_s2
        ),
        "required_force_N": float(required_force_N),
        "acceleration_margin": float(acceleration_margin),
        "thrust_margin": float(thrust_margin),
        "observation_duration_s": float(observation_duration_s),
        "hold_delta_v_m_s": float(hold_delta_v_m_s),
        "hold_impulse_Ns": float(hold_impulse_Ns),
        "hold_propellant_kg": float(hold_propellant_kg),
        "hold_impulse_fraction": float(hold_impulse_fraction),
        "hold_impulse_percent": float(
            100.0 * hold_impulse_fraction
        ),
        "is_thrust_sufficient": bool(thrust_margin >= 1.0),
    }


def get_position(X_grid, sat_index):
    """
    Возвращает массив положений одного ведомого аппарата:
        rho_i(t) = [x_i(t), y_i(t), z_i(t)].
    """

    start = sat_index * NX_ONE

    x = X_grid[:, start + 0]
    y = X_grid[:, start + 2]
    z = X_grid[:, start + 4]

    return np.column_stack([x, y, z])


def get_velocity(X_grid, sat_index):
    """
    Возвращает массив скоростей одного ведомого аппарата:
        v_i(t) = [vx_i(t), vy_i(t), vz_i(t)].
    """

    start = sat_index * NX_ONE

    vx = X_grid[:, start + 1]
    vy = X_grid[:, start + 3]
    vz = X_grid[:, start + 5]

    return np.column_stack([vx, vy, vz])


def compute_basic_errors(solution, config):
    """
    Считает ошибки положения и скорости для трёх ведомых аппаратов.
    """

    X_grid = solution["X"]
    time_grid = solution["time_grid"]

    targets = np.array(
        config["formation"]["targets_m"],
        dtype=float,
    )

    errors = {}

    for i in range(N_SAT):
        rho_i = get_position(X_grid, i)
        v_i = get_velocity(X_grid, i)

        e_rho_i = rho_i - targets[i, :]
        e_norm_i = np.linalg.norm(e_rho_i, axis=1)
        v_norm_i = np.linalg.norm(v_i, axis=1)

        errors[f"sat_{i + 1}"] = {
            "e_rho": e_rho_i,
            "e_norm": e_norm_i,
            "v_norm": v_norm_i,
        }

    obs_mask = time_grid >= config["time"]["t_obs_s"]

    all_e_obs = []
    all_v_obs = []

    for i in range(N_SAT):
        all_e_obs.append(
            errors[f"sat_{i + 1}"]["e_norm"][obs_mask]
        )
        all_v_obs.append(
            errors[f"sat_{i + 1}"]["v_norm"][obs_mask]
        )

    all_e_obs = np.concatenate(all_e_obs)
    all_v_obs = np.concatenate(all_v_obs)

    metrics = {
        "max_position_error_obs_m": float(
            np.max(all_e_obs)
        ),
        "rms_position_error_obs_m": float(
            np.sqrt(np.mean(all_e_obs**2))
        ),
        "max_velocity_norm_obs_m_s": float(
            np.max(all_v_obs)
        ),
        "rms_velocity_norm_obs_m_s": float(
            np.sqrt(np.mean(all_v_obs**2))
        ),
    }

    return errors, metrics


def compute_control_metrics(solution, config):
    """
    Считает характеристики управления и двигательной установки:
        - максимальное управляющее ускорение;
        - максимальную физическую силу тяги;
        - долю насыщения по тяге;
        - долю насыщения по скоростям поворота;
        - накопленную характеристическую скорость;
        - полный импульс;
        - оценку расхода топлива.
    """

    U_grid = solution["U"]

    time_grid = np.asarray(
        solution["time_grid"],
        dtype=float,
    )
    control_time_grid = np.asarray(
        solution["control_time_grid"],
        dtype=float,
    )
    dt_grid = np.diff(time_grid)

    if len(dt_grid) != U_grid.shape[0]:
        raise ValueError(
            "Число интервалов управления не совпадает "
            "с числом шагов времени."
        )

    propulsion = get_propulsion_parameters(config)

    f_max = float(
        config["control"]["f_max_m_s2"]
    )
    omega_psi_max = float(
        config["control"]["omega_psi_max_rad_s"]
    )
    omega_theta_max = float(
        config["control"]["omega_theta_max_rad_s"]
    )

    mass_kg = propulsion["mass_kg"]
    specific_impulse_s = propulsion["specific_impulse_s"]
    total_impulse_Ns = propulsion["total_impulse_Ns"]
    g0_m_s2 = propulsion["g0_m_s2"]

    t_obs_s = float(config["time"]["t_obs_s"])
    obs_mask = control_time_grid >= t_obs_s

    saturation_level = 0.999

    metrics = {}

    for i in range(N_SAT):
        start = i * NU_ONE

        f = U_grid[:, start + 0]
        omega_psi = U_grid[:, start + 1]
        omega_theta = U_grid[:, start + 2]

        force_N = mass_kg * f

        delta_v_m_s = np.sum(f * dt_grid)
        total_impulse_Ns_used = np.sum(
            force_N * dt_grid
        )
        propellant_consumption_kg = (
            total_impulse_Ns_used
            / (specific_impulse_s * g0_m_s2)
        )

        observation_delta_v_m_s = np.sum(
            f[obs_mask] * dt_grid[obs_mask]
        )
        observation_impulse_Ns = np.sum(
            force_N[obs_mask] * dt_grid[obs_mask]
        )
        observation_propellant_kg = (
            observation_impulse_Ns
            / (specific_impulse_s * g0_m_s2)
        )

        f_sat_fraction = np.mean(
            f >= saturation_level * f_max
        )
        omega_psi_sat_fraction = np.mean(
            np.abs(omega_psi)
            >= saturation_level * omega_psi_max
        )
        omega_theta_sat_fraction = np.mean(
            np.abs(omega_theta)
            >= saturation_level * omega_theta_max
        )

        metrics[f"sat_{i + 1}"] = {
            "delta_v_m_s": float(delta_v_m_s),
            "observation_delta_v_m_s": float(
                observation_delta_v_m_s
            ),
            "max_f_m_s2": float(np.max(f)),
            "max_force_N": float(np.max(force_N)),
            "f_saturation_percent": float(
                100.0 * f_sat_fraction
            ),
            "total_impulse_Ns": float(
                total_impulse_Ns_used
            ),
            "observation_impulse_Ns": float(
                observation_impulse_Ns
            ),
            "propellant_consumption_kg": float(
                propellant_consumption_kg
            ),
            "observation_propellant_kg": float(
                observation_propellant_kg
            ),
            "total_impulse_fraction": float(
                total_impulse_Ns_used / total_impulse_Ns
            ),
            "total_impulse_percent": float(
                100.0
                * total_impulse_Ns_used
                / total_impulse_Ns
            ),
            "max_abs_omega_psi_rad_s": float(
                np.max(np.abs(omega_psi))
            ),
            "omega_psi_saturation_percent": float(
                100.0 * omega_psi_sat_fraction
            ),
            "max_abs_omega_theta_rad_s": float(
                np.max(np.abs(omega_theta))
            ),
            "omega_theta_saturation_percent": float(
                100.0 * omega_theta_sat_fraction
            ),
        }

    return metrics


def save_solution_npz(solution, config, results_dir):
    """
    Сохраняет полное решение в бинарном формате NumPy.
    Этот файл удобно использовать для повторного построения
    графиков без нового решения NLP.
    """

    path = results_dir / "solution.npz"

    np.savez(
        path,
        X=solution["X"],
        U=solution["U"],
        time_grid=solution["time_grid"],
        control_time_grid=solution["control_time_grid"],
        objective=np.array([solution["objective"]]),
        status=np.array([solution["status"]]),
        X0=np.array(
            config["initial_state"]["X0"],
            dtype=float,
        ),
        targets_m=np.array(
            config["formation"]["targets_m"],
            dtype=float,
        ),
    )

    return path


def save_trajectories_csv(solution, results_dir):
    """
    Сохраняет координаты, скорости и углы ориентации
    трёх ведомых аппаратов.
    """

    X_grid = solution["X"]
    time_grid = solution["time_grid"]

    header = ["t"]

    for i in range(N_SAT):
        number = i + 1

        header.extend(
            [
                f"x{number}",
                f"vx{number}",
                f"y{number}",
                f"vy{number}",
                f"z{number}",
                f"vz{number}",
                f"psi{number}",
                f"theta{number}",
            ]
        )

    data = np.column_stack([time_grid, X_grid])

    path = results_dir / "trajectories.csv"

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    return path


def save_controls_csv(solution, config, results_dir):
    """
    Сохраняет управления:
        f_i, F_i, omega_psi_i, omega_theta_i.

    Здесь f_i -- модуль управляющего ускорения, м/с^2,
    F_i = m f_i -- соответствующая сила тяги, Н.
    """

    U_grid = solution["U"]
    control_time_grid = solution["control_time_grid"]

    mass_kg = get_propulsion_parameters(
        config
    )["mass_kg"]

    header = ["t"]
    columns = [control_time_grid]

    for i in range(N_SAT):
        number = i + 1
        start = i * NU_ONE

        f = U_grid[:, start + 0]
        omega_psi = U_grid[:, start + 1]
        omega_theta = U_grid[:, start + 2]

        force_N = mass_kg * f

        header.extend(
            [
                f"f{number}",
                f"F{number}_N",
                f"omega_psi{number}",
                f"omega_theta{number}",
            ]
        )

        columns.extend(
            [
                f,
                force_N,
                omega_psi,
                omega_theta,
            ]
        )

    data = np.column_stack(columns)

    path = results_dir / "controls.csv"

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    return path


def save_errors_csv(solution, errors, results_dir):
    """
    Сохраняет ошибки положения и нормы ошибок.
    """

    time_grid = solution["time_grid"]

    header = ["t"]
    columns = [time_grid]

    for i in range(N_SAT):
        number = i + 1

        e_rho = errors[f"sat_{number}"]["e_rho"]
        e_norm = errors[f"sat_{number}"]["e_norm"]
        v_norm = errors[f"sat_{number}"]["v_norm"]

        header.extend(
            [
                f"ex{number}",
                f"ey{number}",
                f"ez{number}",
                f"e_norm{number}",
                f"v_norm{number}",
            ]
        )

        columns.extend(
            [
                e_rho[:, 0],
                e_rho[:, 1],
                e_rho[:, 2],
                e_norm,
                v_norm,
            ]
        )

    data = np.column_stack(columns)

    path = results_dir / "errors.csv"

    np.savetxt(
        path,
        data,
        delimiter=",",
        header=",".join(header),
        comments="",
    )

    return path


def save_solver_summary(
    solution,
    config,
    error_metrics,
    control_metrics,
    propulsion_feasibility,
    results_dir,
):
    """
    Сохраняет краткую сводку по решению.
    """

    summary = {
        "status": solution["status"],
        "objective": float(solution["objective"]),
        "time": {
            "T_s": config["time"]["T_s"],
            "t_obs_s": config["time"]["t_obs_s"],
            "N": config["time"]["N"],
            "dt_s": config["time"]["dt_s"],
        },
        "error_metrics": error_metrics,
        "control_metrics": control_metrics,
        "propulsion_feasibility": propulsion_feasibility,
    }

    path = results_dir / "solver_summary.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return path, summary


def print_summary(summary):
    """
    Печатает основную информацию после решения.
    """

    print()
    print("Решение завершено.")
    print(f"Статус: {summary['status']}")
    print(f"Функционал: {summary['objective']:.8e}")

    print()
    print("Ошибки на участке наблюдения:")

    print(
        "max ||e|| = "
        f"{summary['error_metrics']['max_position_error_obs_m']:.6f} м"
    )
    print(
        "RMS ||e|| = "
        f"{summary['error_metrics']['rms_position_error_obs_m']:.6f} м"
    )
    print(
        "max ||v|| = "
        f"{summary['error_metrics']['max_velocity_norm_obs_m_s']:.6f} м/с"
    )
    print(
        "RMS ||v|| = "
        f"{summary['error_metrics']['rms_velocity_norm_obs_m_s']:.6f} м/с"
    )

    feasibility = summary["propulsion_feasibility"]

    print()
    print("Проверка двигательной установки:")

    print(
        "Требуемое ускорение для удержания z=d: "
        f"{feasibility['required_acceleration_m_s2']:.8e} м/с^2"
    )
    print(
        "Требуемая сила тяги для удержания z=d: "
        f"{feasibility['required_force_N']:.6f} Н"
    )
    print(
        f"Запас по тяге: "
        f"{feasibility['thrust_margin']:.6f}"
    )
    print(
        "Оценочный импульс удержания "
        "за рабочий интервал: "
        f"{feasibility['hold_impulse_Ns']:.6f} Н·с"
    )
    print(
        "Оценочный расход топлива "
        "за рабочий интервал: "
        f"{feasibility['hold_propellant_kg']:.6f} кг"
    )
    print(
        "Доля полного импульса "
        "на один рабочий интервал: "
        f"{feasibility['hold_impulse_percent']:.3f} %"
    )
    print(
        "Тяги достаточно: "
        f"{'да' if feasibility['is_thrust_sufficient'] else 'нет'}"
    )

    print()
    print("Затраты управления:")

    for i in range(N_SAT):
        key = f"sat_{i + 1}"
        sat_metrics = summary["control_metrics"][key]

        print(f"Аппарат {i + 1}:")

        print(
            f"  Delta v = "
            f"{sat_metrics['delta_v_m_s']:.6f} м/с"
        )
        print(
            "  Delta v на участке наблюдения = "
            f"{sat_metrics['observation_delta_v_m_s']:.6f} м/с"
        )
        print(
            "  max f = "
            f"{sat_metrics['max_f_m_s2']:.8e} м/с^2"
        )
        print(
            f"  max F = "
            f"{sat_metrics['max_force_N']:.6f} Н"
        )
        print(
            "  полный импульс = "
            f"{sat_metrics['total_impulse_Ns']:.6f} Н·с"
        )
        print(
            "  расход топлива = "
            f"{sat_metrics['propellant_consumption_kg']:.6f} кг"
        )
        print(
            "  доля полного импульса = "
            f"{sat_metrics['total_impulse_percent']:.3f} %"
        )
        print(
            "  насыщение по f = "
            f"{sat_metrics['f_saturation_percent']:.2f} %"
        )
        print(
            "  omega_psi_sat = "
            f"{sat_metrics['omega_psi_saturation_percent']:.2f} %"
        )
        print(
            "  omega_theta_sat = "
            f"{sat_metrics['omega_theta_saturation_percent']:.2f} %"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Решение задачи оптимального управления "
            "для квадратной конфигурации КА."
        )
    )

    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help=(
            "Число интервалов сетки. "
            "Например, для теста можно поставить --N 60."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Максимальное число итераций IPOPT.",
    )

    parser.add_argument(
        "--print-level",
        type=int,
        default=None,
        help=(
            "Уровень вывода IPOPT. "
            "Для тихого запуска можно поставить 0."
        ),
    )

    parser.add_argument(
        "--w-prep",
        type=float,
        default=None,
        help=(
            "Вес штрафа на подготовительном "
            "участке [0, t_obs)."
        ),
    )

    parser.add_argument(
        "--eps-thrust",
        type=float,
        default=None,
        help="Малый штраф за использование тяги.",
    )

    parser.add_argument(
        "--eps-angle-rate",
        type=float,
        default=None,
        help=(
            "Малый штраф за скорости "
            "изменения углов ориентации."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = build_config()
    config = update_config_from_args(config, args)

    propulsion = get_propulsion_parameters(config)
    propulsion_feasibility = compute_propulsion_feasibility(
        config
    )

    project = ProjectConfig()
    make_result_dirs(project)

    results_dir = Path(
        config["project"]["results_dir"]
    )
    figures_dir = Path(
        config["project"]["figures_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = results_dir / "config.json"
    save_config(config, config_path)

    print("Конфигурация сохранена:")
    print(config_path)

    print()
    print("Запуск решения задачи оптимального управления...")
    print(f"N = {config['time']['N']}")
    print(f"dt = {config['time']['dt_s']:.3f} с")
    print(f"T = {config['time']['T_s']:.3f} с")
    print(f"m = {propulsion['mass_kg']:.3f} кг")
    print(f"F_max = {propulsion['F_max_N']:.6f} Н")
    print(
        f"f_max = "
        f"{propulsion['f_max_m_s2']:.8e} м/с^2"
    )

    solution = solve_ocp(config)

    errors, error_metrics = compute_basic_errors(
        solution,
        config,
    )
    control_metrics = compute_control_metrics(
        solution,
        config,
    )

    solution_path = save_solution_npz(
        solution,
        config,
        results_dir,
    )
    trajectories_path = save_trajectories_csv(
        solution,
        results_dir,
    )
    controls_path = save_controls_csv(
        solution,
        config,
        results_dir,
    )
    errors_path = save_errors_csv(
        solution,
        errors,
        results_dir,
    )

    summary_path, summary = save_solver_summary(
        solution=solution,
        config=config,
        error_metrics=error_metrics,
        control_metrics=control_metrics,
        propulsion_feasibility=propulsion_feasibility,
        results_dir=results_dir,
    )

    print_summary(summary)

    print()
    print("Сохранённые файлы:")
    print(solution_path)
    print(trajectories_path)
    print(controls_path)
    print(errors_path)
    print(summary_path)


if __name__ == "__main__":
    main()