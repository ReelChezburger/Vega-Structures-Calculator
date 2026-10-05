#I chatGPT'd this lol, just testing if I'm thinking about it right

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


# ============================================================
# INPUTS
# ============================================================

L = 10.0                 # vehicle length [m]
N = 2000                 # beam discretization

# ------------------------------------------------------------
# Components
#
# mass_distribution:
#   array of [relative_position, mass_fraction]
#
# relative_position goes from 0 to 1:
#   0 = front of component
#   1 = rear of component
#
# mass_fraction must integrate to 1 over the component.
# ------------------------------------------------------------

components = [
    {
        "name": "Nose cone",
        "mass": 100.0,          # kg
        "x_front": 0.0,         # m
        "length": 1.0,          # m

        # Solid tangent-ogive mass distribution
        #
        # Mass per unit length is proportional to
        # cross-sectional area: A(x) = pi * r(x)^2
        #
        # r(x) is generated from the ogive geometry.
        #
        # Replace this with a function or something for the real code
        "mass_distribution": np.array([
            [0.0, 0.0],
            [0.1, 0.012],
            [0.2, 0.050],
            [0.3, 0.116],
            [0.4, 0.215],
            [0.5, 0.350],
            [0.6, 0.520],
            [0.7, 0.710],
            [0.8, 0.870],
            [0.9, 0.970],
            [1.0, 1.000]
        ])
    },

    {
        "name": "Forward tank",
        "mass": 500.0,           # kg
        "x_front": 1.0,          # m
        "length": 3.0,           # m

        # Uniform mass distribution
        "mass_distribution": np.array([
            [0.0, 1.0],
            [1.0, 1.0]
        ])
    },

    {
        "name": "Avionics",
        "mass": 100.0,
        "x_front": 4.0,
        "length": 0.5,

        # Concentrated toward front in this example
        "mass_distribution": np.array([
            [0.0, 1.5],
            [0.5, 0.5],
            [1.0, 0.1]
        ])
    },

    {
        "name": "Aft tank",
        "mass": 700.0,
        "x_front": 5.0,
        "length": 3.0,

        "mass_distribution": np.array([
            [0.0, 0.5],
            [1.0, 1.5]
        ])
    }
]


# ------------------------------------------------------------
# External point forces
#
# Positive = upward / + bending direction
# Negative = downward / - bending direction
#
# Nose force should be negative.
# Fin force should be positive.
# ------------------------------------------------------------

forces = [
    {
        "name": "Nose",
        "force": -2.0,       # N
        "x": 0.0             # m
    },

    {
        "name": "Fins",
        "force": 12.0,       # N
        "x": L               # m
    }
]


# ============================================================
# BUILD VEHICLE MASS DISTRIBUTION
# ============================================================

x = np.linspace(0.0, L, N)
dx = x[1] - x[0]

lambda_x = np.zeros_like(x)       # kg/m


for component in components:

    x_front = component["x_front"]
    length = component["length"]
    mass = component["mass"]

    distribution = component["mass_distribution"]

    # Local normalized position
    s = distribution[:, 0]

    # User-supplied distribution shape
    shape = distribution[:, 1]

    # Convert to actual x locations
    x_component = x_front + s * length

    # Normalize distribution so integral = 1
    normalization = np.trapezoid(shape, x_component)
    shape = shape / normalization

    # Interpolate component's kg/m distribution
    component_lambda = np.interp(
        x,
        x_component,
        mass * shape,
        left=0.0,
        right=0.0
    )

    lambda_x += component_lambda


# ============================================================
# VEHICLE MASS PROPERTIES
# ============================================================

total_mass = np.trapezoid(lambda_x, x)

x_CG = (
    np.trapezoid(x * lambda_x, x)
    / total_mass
)

I_CG = np.trapezoid(
    (x - x_CG)**2 * lambda_x,
    x
)

print(f"Total mass = {total_mass:.3f} kg")
print(f"CG location = {x_CG:.4f} m")
print(f"I_CG = {I_CG:.4f} kg m^2")


# ============================================================
# EXTERNAL FORCE RESULTANTS
# ============================================================

F_total = sum(force["force"] for force in forces)

moment_CG = sum(
    force["force"] * (force["x"] - x_CG)
    for force in forces
)


# ============================================================
# VEHICLE TRANSLATIONAL AND ANGULAR ACCELERATION
# ============================================================

a_CG = F_total / total_mass

alpha = moment_CG / I_CG

print(f"Net force = {F_total:.4f} N")
print(f"a_CG = {a_CG:.6f} m/s^2")
print(f"alpha = {alpha:.6f} rad/s^2")


# ============================================================
# ACCELERATION AT EVERY LOCATION
# ============================================================

a_x = a_CG + alpha * (x - x_CG)


# ============================================================
# DISTRIBUTED INERTIAL LOAD
#
# q = -lambda * a
#
# Positive q = upward
# Negative q = downward
# ============================================================

q_x = -lambda_x * a_x


# ============================================================
# SHEAR FORCE
#
# V(x) = sum(point forces to the left)
#        + integral(q dx)
#
# We calculate the distributed component first.
# ============================================================

V_x = cumulative_trapezoid(
    q_x,
    x,
    initial=0.0
)


# Add point-force jumps
for force in forces:

    force_x = force["x"]
    force_value = force["force"]

    mask = x >= force_x

    V_x[mask] += force_value


# ============================================================
# BENDING MOMENT
#
# dM/dx = V
# M(0) = 0
# ============================================================

M_x = cumulative_trapezoid(
    V_x,
    x,
    initial=0.0
)


# ============================================================
# EQUILIBRIUM CHECKS
# ============================================================

print("\n--- Equilibrium checks ---")

print(f"V(left)  = {V_x[0]:.6e} N")
print(f"V(right) = {V_x[-1]:.6e} N")

print(f"M(left)  = {M_x[0]:.6e} N m")
print(f"M(right) = {M_x[-1]:.6e} N m")


# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axes[0].plot(x, lambda_x)
axes[0].set_ylabel("Mass / length\n[kg/m]")
axes[0].grid(True)

axes[1].plot(x, a_x)
axes[1].set_ylabel("Acceleration\n[m/s²]")
axes[1].grid(True)

axes[2].plot(x, V_x)
axes[2].axhline(0.0, linewidth=0.8)
axes[2].set_ylabel("Shear\n[N]")
axes[2].grid(True)

axes[3].plot(x, M_x)
axes[3].axhline(0.0, linewidth=0.8)
axes[3].set_ylabel("Moment\n[N m]")
axes[3].set_xlabel("x [m]")
axes[3].grid(True)

plt.tight_layout()
plt.show()