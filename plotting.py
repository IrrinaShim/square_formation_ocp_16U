import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import ProjectConfig
from dynamics import N_SAT, NX_ONE, NU_ONE


SAT_COLORS = {
    0: "tab:blue",
    1: "tab:orange",
    2: "tab:green",
}

REF_COLOR = "black"
TARGET_SQUARE_COLOR = "0.45"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_results(results_dir):
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
        "targets_m": data["targets_m"],
        "objective": float(data["objective"][0]),
        "status": str(data["status"][0]),
    }

    return config, solution


def get_propulsion_parameters(config):
    """
    Извлекает параметры двигательной установки и проверяет согласованность
    f_max = F_max / m.
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


def get_hold_requirements(config):
    """
    Возвращает ускорение и силу, необходимые для удержания z = d.
    """

    propulsion = get_propulsion_parameters(config)

    q = float(config["orbit"]["q"])
    d_m = float(config["formation"]["d_m"])

    required_acceleration_m_s2 = q**2 * d_m
    required_force_N = propulsion["mass_kg"] * required_acceleration_m_s2

    return required_acceleration_m_s2, required_force_N


def prepare_figure_dirs(results_dir):
    figures_dir = Path(results_dir) / "figures"

    dirs = {
        "root": figures_dir,
        "controls": figures_dir / "controls",
        "states": figures_dir / "states",
        "shape": figures_dir / "shape",
        "projections": figures_dir / "projections",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def save_png(fig, directory, name):
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def get_position(X_grid, sat_index):
    start = sat_index * NX_ONE

    x = X_grid[:, start + 0]
    y = X_grid[:, start + 2]
    z = X_grid[:, start + 4]

    return x, y, z


def get_velocity(X_grid, sat_index):
    start = sat_index * NX_ONE

    vx = X_grid[:, start + 1]
    vy = X_grid[:, start + 3]
    vz = X_grid[:, start + 5]

    return vx, vy, vz


def get_angles(X_grid, sat_index):
    start = sat_index * NX_ONE

    psi = X_grid[:, start + 6]
    theta = X_grid[:, start + 7]

    return psi, theta


def get_control(U_grid, sat_index):
    start = sat_index * NU_ONE

    f = U_grid[:, start + 0]
    omega_psi = U_grid[:, start + 1]
    omega_theta = U_grid[:, start + 2]

    return f, omega_psi, omega_theta


def set_equal_3d_axes(ax, x_values, y_values, z_values):
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    z_min, z_max = np.min(z_values), np.max(z_values)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    if max_range == 0:
        max_range = 1.0

    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def set_equal_2d_axes(ax, x_values, y_values, margin_fraction=0.05):
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    max_range = max(
        x_max - x_min,
        y_max - y_min,
    )

    if max_range == 0:
        max_range = 1.0

    half = 0.5 * max_range * (1.0 + margin_fraction)

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_aspect("equal", adjustable="box")


def get_square_vertices(targets):
    vertices = np.vstack(
        [
            np.zeros((1, 3)),
            targets,
        ]
    )

    square_order = [0, 1, 3, 2, 0]

    return vertices, square_order


def collect_all_phase_values(config, solution):
    X_grid = solution["X"]
    targets = np.array(config["formation"]["targets_m"], dtype=float)

    all_x = [0.0]
    all_y = [0.0]
    all_z = [0.0]

    for i in range(N_SAT):
        x, y, z = get_position(X_grid, i)

        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
        all_z.extend(z.tolist())

        all_x.append(targets[i, 0])
        all_y.append(targets[i, 1])
        all_z.append(targets[i, 2])

    return np.array(all_x), np.array(all_y), np.array(all_z)


def make_vertical_satellite_axes(ylabel, figsize=(7.0, 8.0), sharex=True):
    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=figsize,
        sharex=sharex,
    )

    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend_handles_labels = None

    axes[-1].set_xlabel("$t$, с")

    return fig, axes


def plot_phase_xyz(config, solution, dirs):
    X_grid = solution["X"]
    targets = np.array(config["formation"]["targets_m"], dtype=float)

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    vertices, square_order = get_square_vertices(targets)
    all_x, all_y, all_z = collect_all_phase_values(config, solution)

    ax.plot(
        vertices[square_order, 0],
        vertices[square_order, 1],
        vertices[square_order, 2],
        linestyle="--",
        linewidth=1.0,
        color=TARGET_SQUARE_COLOR,
        label="требуемый квадрат",
        zorder=1,
    )

    ax.scatter(
        0.0,
        0.0,
        0.0,
        marker="o",
        s=55,
        color=REF_COLOR,
        label="опорный КА",
        zorder=5,
    )

    for i in range(N_SAT):
        color = SAT_COLORS[i]
        x, y, z = get_position(X_grid, i)

        ax.plot(
            x,
            y,
            z,
            linewidth=2.0,
            color=color,
            label=f"траектория КА {i + 1}",
            zorder=4,
        )

        ax.scatter(
            targets[i, 0],
            targets[i, 1],
            targets[i, 2],
            marker="x",
            s=80,
            color=color,
            linewidths=2.0,
            label=f"цель КА {i + 1}",
            zorder=6,
        )

    ax.set_xlabel("$x$, м")
    ax.set_ylabel("$y$, м")
    ax.set_zlabel("$z$, м")

    ax.grid(True)
    ax.legend(loc="best")

    set_equal_3d_axes(ax, all_x, all_y, all_z)

    fig.tight_layout()

    return save_png(fig, dirs["root"], "phase_xyz")


def plot_phase_yz(config, solution, dirs):
    X_grid = solution["X"]
    targets = np.array(config["formation"]["targets_m"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    vertices, square_order = get_square_vertices(targets)
    _, all_y, all_z = collect_all_phase_values(config, solution)

    ax.plot(
        vertices[square_order, 1],
        vertices[square_order, 2],
        linestyle="--",
        linewidth=1.0,
        color=TARGET_SQUARE_COLOR,
        label="требуемый квадрат",
        zorder=1,
    )

    ax.scatter(
        0.0,
        0.0,
        marker="o",
        s=55,
        color=REF_COLOR,
        label="опорный КА",
        zorder=5,
    )

    for i in range(N_SAT):
        color = SAT_COLORS[i]
        _, y, z = get_position(X_grid, i)

        ax.plot(
            y,
            z,
            linewidth=2.0,
            color=color,
            label=f"траектория КА {i + 1}",
            zorder=4,
        )

        ax.scatter(
            targets[i, 1],
            targets[i, 2],
            marker="x",
            s=80,
            color=color,
            linewidths=2.0,
            label=f"цель КА {i + 1}",
            zorder=6,
        )

    ax.set_xlabel("$y$, м")
    ax.set_ylabel("$z$, м")

    ax.grid(True)
    ax.legend(loc="best")

    set_equal_2d_axes(ax, all_y, all_z)

    fig.tight_layout()

    return save_png(fig, dirs["root"], "phase_yz")


def plot_x_t(solution, dirs):
    X_grid = solution["X"]
    t = solution["time_grid"]

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        x, _, _ = get_position(X_grid, i)

        ax.plot(
            t,
            x,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.set_ylabel("$x$, м")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["root"], "x_t")


def plot_position_errors(config, solution, dirs):
    X_grid = solution["X"]
    t = solution["time_grid"]

    targets = np.array(config["formation"]["targets_m"], dtype=float)
    rho_tol = float(config["formation"]["rho_tol_m"])
    t_obs = float(config["time"]["t_obs_s"])

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        x, y, z = get_position(X_grid, i)
        rho = np.column_stack([x, y, z])

        e = rho - targets[i, :]
        e_norm = np.linalg.norm(e, axis=1)

        ax.plot(
            t,
            e_norm,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.axhline(
            rho_tol,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$\Delta d_{\max}$",
        )

        ax.axvline(
            t_obs,
            linestyle=":",
            linewidth=1.0,
            color="black",
            label=r"$t_{\mathrm{obs}}$",
        )

        ax.set_ylabel(r"$\|\rho_i-\rho_i^*\|$, м")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["root"], "position_errors")


def plot_projections(config, solution, dirs):
    X_grid = solution["X"]
    targets = np.array(config["formation"]["targets_m"], dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 14.0),
    )

    projections = [
        ("$x$, м", "$y$, м", 0, 1),
        ("$x$, м", "$z$, м", 0, 2),
        ("$y$, м", "$z$, м", 1, 2),
    ]

    vertices, square_order = get_square_vertices(targets)

    for ax, (xlabel, ylabel, a, b) in zip(axes, projections):
        ax.plot(
            vertices[square_order, a],
            vertices[square_order, b],
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label="требуемый квадрат",
            zorder=1,
        )

        ax.scatter(
            0.0,
            0.0,
            marker="o",
            s=35,
            color=REF_COLOR,
            label="опорный КА",
            zorder=5,
        )

        all_a = [0.0]
        all_b = [0.0]

        for i in range(N_SAT):
            x, y, z = get_position(X_grid, i)
            coords = np.column_stack([x, y, z])
            color = SAT_COLORS[i]

            ax.plot(
                coords[:, a],
                coords[:, b],
                linewidth=1.6,
                color=color,
                label=f"траектория КА {i + 1}",
                zorder=4,
            )

            ax.scatter(
                targets[i, a],
                targets[i, b],
                marker="x",
                s=65,
                color=color,
                linewidths=2.0,
                label=f"цель КА {i + 1}",
                zorder=6,
            )

            all_a.extend(coords[:, a].tolist())
            all_b.extend(coords[:, b].tolist())
            all_a.append(targets[i, a])
            all_b.append(targets[i, b])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(loc="best")

        set_equal_2d_axes(
            ax,
            np.array(all_a),
            np.array(all_b),
        )

    fig.tight_layout()

    return save_png(fig, dirs["projections"], "projections_xy_xz_yz")


def plot_state_coordinates(solution, dirs):
    X_grid = solution["X"]
    t = solution["time_grid"]

    coordinate_info = [
        ("$y$, м", 2, "y_t"),
        ("$z$, м", 4, "z_t"),
    ]

    saved_paths = []

    for ylabel, local_index, filename in coordinate_info:
        fig, axes = plt.subplots(
            N_SAT,
            1,
            figsize=(7.0, 8.0),
            sharex=True,
        )

        for i, ax in enumerate(axes):
            start = i * NX_ONE

            ax.plot(
                t,
                X_grid[:, start + local_index],
                linewidth=1.5,
                color=SAT_COLORS[i],
                label=f"КА {i + 1}",
            )

            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc="best")

        axes[-1].set_xlabel("$t$, с")

        fig.tight_layout()

        saved_paths.append(save_png(fig, dirs["states"], filename))

    return saved_paths


def plot_state_velocities(solution, dirs):
    X_grid = solution["X"]
    t = solution["time_grid"]

    velocity_info = [
        ("$v_x$, м/с", 1, "vx_t"),
        ("$v_y$, м/с", 3, "vy_t"),
        ("$v_z$, м/с", 5, "vz_t"),
    ]

    saved_paths = []

    for ylabel, local_index, filename in velocity_info:
        fig, axes = plt.subplots(
            N_SAT,
            1,
            figsize=(7.0, 8.0),
            sharex=True,
        )

        for i, ax in enumerate(axes):
            start = i * NX_ONE

            ax.plot(
                t,
                X_grid[:, start + local_index],
                linewidth=1.5,
                color=SAT_COLORS[i],
                label=f"КА {i + 1}",
            )

            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend(loc="best")

        axes[-1].set_xlabel("$t$, с")

        fig.tight_layout()

        saved_paths.append(save_png(fig, dirs["states"], filename))

    return saved_paths


def compute_shape_arrays(config, solution):
    X_grid = solution["X"]

    d = float(config["formation"]["d_m"])
    diag_ref = np.sqrt(2.0) * d

    n_points = X_grid.shape[0]

    rho0 = np.zeros((n_points, 3))

    positions = [rho0]

    for i in range(N_SAT):
        x, y, z = get_position(X_grid, i)
        positions.append(np.column_stack([x, y, z]))

    rho0, rho1, rho2, rho3 = positions

    def dist(a, b):
        return np.linalg.norm(a - b, axis=1)

    shape = {
        "L01": dist(rho0, rho1),
        "L02": dist(rho0, rho2),
        "L13": dist(rho1, rho3),
        "L23": dist(rho2, rho3),
        "D03": dist(rho0, rho3),
        "D12": dist(rho1, rho2),
        "d_ref": d,
        "diag_ref": diag_ref,
    }

    return shape


def plot_shape_lengths(config, solution, dirs):
    t = solution["time_grid"]
    t_obs = float(config["time"]["t_obs_s"])

    shape = compute_shape_arrays(config, solution)

    side_names = ["L01", "L02", "L13", "L23"]

    side_colors = {
        "L01": "tab:blue",
        "L02": "tab:orange",
        "L13": "tab:green",
        "L23": "tab:red",
    }

    fig, axes = plt.subplots(
        len(side_names),
        1,
        figsize=(7.0, 9.0),
        sharex=True,
    )

    for ax, name in zip(axes, side_names):
        ax.plot(
            t,
            shape[name],
            linewidth=1.3,
            color=side_colors[name],
            label=f"${name}$",
        )

        ax.axhline(
            shape["d_ref"],
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label="$d$",
        )

        ax.axvline(
            t_obs,
            linestyle=":",
            linewidth=1.0,
            color="black",
            label=r"$t_{\mathrm{obs}}$",
        )

        ax.set_ylabel("м")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["shape"], "side_lengths")


def plot_shape_diagonals(config, solution, dirs):
    t = solution["time_grid"]
    t_obs = float(config["time"]["t_obs_s"])

    shape = compute_shape_arrays(config, solution)

    diagonal_names = ["D03", "D12"]

    diagonal_colors = {
        "D03": "tab:purple",
        "D12": "tab:brown",
    }

    fig, axes = plt.subplots(
        len(diagonal_names),
        1,
        figsize=(7.0, 5.8),
        sharex=True,
    )

    for ax, name in zip(axes, diagonal_names):
        ax.plot(
            t,
            shape[name],
            linewidth=1.5,
            color=diagonal_colors[name],
            label=f"${name}$",
        )

        ax.axhline(
            shape["diag_ref"],
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$\sqrt{2}d$",
        )

        ax.axvline(
            t_obs,
            linestyle=":",
            linewidth=1.0,
            color="black",
            label=r"$t_{\mathrm{obs}}$",
        )

        ax.set_ylabel("м")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["shape"], "diagonals")


def plot_shape_relative_errors(config, solution, dirs):
    t = solution["time_grid"]
    t_obs = float(config["time"]["t_obs_s"])

    shape = compute_shape_arrays(config, solution)

    d = shape["d_ref"]
    diag_ref = shape["diag_ref"]

    names = ["L01", "L02", "L13", "L23", "D03", "D12"]

    colors = {
        "L01": "tab:blue",
        "L02": "tab:orange",
        "L13": "tab:green",
        "L23": "tab:red",
        "D03": "tab:purple",
        "D12": "tab:brown",
    }

    fig, axes = plt.subplots(
        len(names),
        1,
        figsize=(7.0, 12.0),
        sharex=True,
    )

    for ax, name in zip(axes, names):
        if name.startswith("L"):
            rel_err = (shape[name] - d) / d
        else:
            rel_err = (shape[name] - diag_ref) / diag_ref

        ax.plot(
            t,
            rel_err,
            linewidth=1.3,
            color=colors[name],
            label=f"${name}$",
        )

        ax.axhline(
            0.0,
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
        )

        ax.axvline(
            t_obs,
            linestyle=":",
            linewidth=1.0,
            color="black",
            label=r"$t_{\mathrm{obs}}$",
        )

        ax.set_ylabel("отн. ошибка")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["shape"], "relative_shape_errors")


def plot_control_acceleration(config, solution, dirs):
    """
    Строит модуль управляющего ускорения f_i.
    """

    U_grid = solution["U"]
    t_u = solution["control_time_grid"]

    f_max = float(config["control"]["f_max_m_s2"])
    required_acceleration, _ = get_hold_requirements(config)

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        f, _, _ = get_control(U_grid, i)

        ax.step(
            t_u,
            f,
            where="post",
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.axhline(
            f_max,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$f_{\max}$",
        )

        if i in (1, 2):
            ax.axhline(
                required_acceleration,
                linestyle=":",
                linewidth=1.2,
                color="tab:red",
                label=r"$q^2d$",
            )

        ax.set_ylabel("$f$, м/с$^2$")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["controls"], "control_acceleration")


def plot_thrust_force(config, solution, dirs):
    """
    Строит физическую силу тяги F_i = m f_i.
    """

    U_grid = solution["U"]
    t_u = solution["control_time_grid"]

    propulsion = get_propulsion_parameters(config)
    _, required_force_N = get_hold_requirements(config)

    mass_kg = propulsion["mass_kg"]
    F_max_N = propulsion["F_max_N"]

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        f, _, _ = get_control(U_grid, i)
        force_N = mass_kg * f

        ax.step(
            t_u,
            force_N,
            where="post",
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.axhline(
            F_max_N,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$F_{\max}$",
        )

        if i in (1, 2):
            ax.axhline(
                required_force_N,
                linestyle=":",
                linewidth=1.2,
                color="tab:red",
                label=r"$F_{z,\mathrm{req}}$",
            )

        ax.set_ylabel("$F$, Н")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["controls"], "thrust")


def plot_angle_rates(config, solution, dirs):
    U_grid = solution["U"]
    t_u = solution["control_time_grid"]

    omega_psi_max = float(config["control"]["omega_psi_max_rad_s"])
    omega_theta_max = float(config["control"]["omega_theta_max_rad_s"])

    saved_paths = []

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        _, omega_psi, _ = get_control(U_grid, i)

        ax.step(
            t_u,
            omega_psi,
            where="post",
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.axhline(
            omega_psi_max,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$\omega_{\psi\max}$",
        )

        ax.axhline(
            -omega_psi_max,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
        )

        ax.set_ylabel(r"$\omega_\psi$, рад/с")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    saved_paths.append(save_png(fig, dirs["controls"], "omega_psi"))

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        _, _, omega_theta = get_control(U_grid, i)

        ax.step(
            t_u,
            omega_theta,
            where="post",
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.axhline(
            omega_theta_max,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$\omega_{\theta\max}$",
        )

        ax.axhline(
            -omega_theta_max,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
        )

        ax.set_ylabel(r"$\omega_\theta$, рад/с")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    saved_paths.append(save_png(fig, dirs["controls"], "omega_theta"))

    return saved_paths


def plot_angles(solution, dirs):
    X_grid = solution["X"]
    t = solution["time_grid"]

    saved_paths = []

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        psi, _ = get_angles(X_grid, i)

        ax.plot(
            t,
            psi,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.set_ylabel(r"$\psi$, рад")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    saved_paths.append(save_png(fig, dirs["controls"], "psi"))

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        _, theta = get_angles(X_grid, i)

        ax.plot(
            t,
            theta,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.set_ylabel(r"$\theta$, рад")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    saved_paths.append(save_png(fig, dirs["controls"], "theta"))

    return saved_paths


def compute_cumulative_propulsion(config, solution):
    """
    Считает накопленные Delta v, импульс и расход топлива на time_grid.
    """

    propulsion = get_propulsion_parameters(config)

    U_grid = solution["U"]
    t = np.asarray(solution["time_grid"], dtype=float)
    dt_grid = np.diff(t)

    if len(dt_grid) != U_grid.shape[0]:
        raise ValueError(
            "Число интервалов управления не совпадает с числом шагов времени."
        )

    histories = []

    for i in range(N_SAT):
        f, _, _ = get_control(U_grid, i)

        delta_v = np.zeros(len(t))
        delta_v[1:] = np.cumsum(f * dt_grid)

        impulse_Ns = propulsion["mass_kg"] * delta_v
        propellant_kg = impulse_Ns / (
            propulsion["specific_impulse_s"] * propulsion["g0_m_s2"]
        )
        impulse_fraction = impulse_Ns / propulsion["total_impulse_Ns"]

        histories.append(
            {
                "delta_v_m_s": delta_v,
                "impulse_Ns": impulse_Ns,
                "propellant_kg": propellant_kg,
                "impulse_fraction": impulse_fraction,
            }
        )

    return histories


def plot_delta_v(config, solution, dirs):
    t = solution["time_grid"]

    histories = compute_cumulative_propulsion(config, solution)

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        delta_v = histories[i]["delta_v_m_s"]

        ax.plot(
            t,
            delta_v,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.set_ylabel(r"$\Delta v$, м/с")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["controls"], "delta_v")


def plot_total_impulse(config, solution, dirs):
    t = solution["time_grid"]
    histories = compute_cumulative_propulsion(config, solution)

    total_impulse_Ns = get_propulsion_parameters(config)["total_impulse_Ns"]

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        impulse_Ns = histories[i]["impulse_Ns"]

        ax.plot(
            t,
            impulse_Ns,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.axhline(
            total_impulse_Ns,
            linestyle="--",
            linewidth=1.0,
            color=TARGET_SQUARE_COLOR,
            label=r"$I_{\Sigma}$",
        )

        ax.set_ylabel("$I$, Н·с")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["controls"], "total_impulse")


def plot_propellant_consumption(config, solution, dirs):
    t = solution["time_grid"]
    histories = compute_cumulative_propulsion(config, solution)

    fig, axes = plt.subplots(
        N_SAT,
        1,
        figsize=(7.0, 8.0),
        sharex=True,
    )

    for i, ax in enumerate(axes):
        propellant_kg = histories[i]["propellant_kg"]

        ax.plot(
            t,
            propellant_kg,
            linewidth=1.5,
            color=SAT_COLORS[i],
            label=f"КА {i + 1}",
        )

        ax.set_ylabel(r"$\Delta m$, кг")
        ax.grid(True)
        ax.legend(loc="best")

    axes[-1].set_xlabel("$t$, с")

    fig.tight_layout()

    return save_png(fig, dirs["controls"], "propellant_consumption")


def plot_all(results_dir=None):
    if results_dir is None:
        results_dir = ProjectConfig().results_dir

    results_dir = Path(results_dir)

    dirs = prepare_figure_dirs(results_dir)

    config, solution = load_results(results_dir)

    saved = []

    saved.append(plot_phase_xyz(config, solution, dirs))
    saved.append(plot_phase_yz(config, solution, dirs))
    saved.append(plot_x_t(solution, dirs))
    saved.append(plot_position_errors(config, solution, dirs))

    saved.append(plot_projections(config, solution, dirs))

    saved.extend(plot_state_coordinates(solution, dirs))
    saved.extend(plot_state_velocities(solution, dirs))

    saved.append(plot_shape_lengths(config, solution, dirs))
    saved.append(plot_shape_diagonals(config, solution, dirs))
    saved.append(plot_shape_relative_errors(config, solution, dirs))

    saved.append(plot_control_acceleration(config, solution, dirs))
    saved.append(plot_thrust_force(config, solution, dirs))
    saved.extend(plot_angle_rates(config, solution, dirs))
    saved.extend(plot_angles(solution, dirs))
    saved.append(plot_delta_v(config, solution, dirs))
    saved.append(plot_total_impulse(config, solution, dirs))
    saved.append(plot_propellant_consumption(config, solution, dirs))

    print()
    print("Графики сохранены в папку:")
    print(dirs["root"])

    print()
    print("Основные графики:")
    for path in [
        dirs["root"] / "phase_xyz.png",
        dirs["root"] / "phase_yz.png",
        dirs["root"] / "x_t.png",
        dirs["root"] / "position_errors.png",
    ]:
        print(path)

    print()
    print("Остальные графики разложены по папкам:")
    print(dirs["controls"])
    print(dirs["states"])
    print(dirs["shape"])
    print(dirs["projections"])


def main():
    plot_all()


if __name__ == "__main__":
    main()
