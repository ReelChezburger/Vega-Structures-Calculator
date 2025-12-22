import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

import math

# Pull constants from vegaParameters csv
with open("parameters.csv", newline="") as f:
    reader = csv.DictReader(f)
    params = next(reader)  # single row

AREF = float(params["AREF"])
ROCKET_LENGTH = float(params["ROCKET_LENGTH"])
CG_DRY = float(params["CG_DRY"])
CG_WET = float(params["CG_WET"])
OF_RATIO = float(params["OF_RATIO"])

FUEL_TANK_LENGTH = float(params["FUEL_TANK_LENGTH"])
FUEL_TANK_POSITION = float(params["FUEL_TANK_POSITION"])

LOX_TANK_LENGTH = float(params["LOX_TANK_LENGTH"])
LOX_TANK_POSITION = float(params["LOX_TANK_POSITION"])

LAUNCH_ALT = 2000/3.281 # m
WIND_GUST = 9 # m/s
N_STATIONS = 100

def rho(h):
    """ 
    Compute air density (kg/m^3) at altitude h (meters)
    using the U.S. Standard Atmosphere 1976 (0–86 km).
    """

    # Define layer base heights (m), base temperatures (K),
    # base pressures (Pa), and lapse rates (K/m)
    layers = [
        (0,     288.15, 101325.0,     -0.0065),
        (11000, 216.65, 22632.06,      0.0000),
        (20000, 216.65, 5474.889,      0.0010),
        (32000, 228.65, 868.019,       0.0028),
        (47000, 270.65, 110.906,       0.0000),
        (51000, 270.65, 66.9389,      -0.0028),
        (71000, 214.65, 3.95639,      -0.0020),
        (84852, 186.87, 0.3734,        0.0000)
    ]

    # Gas constant for air
    R = 287.05287  # J/(kg·K)
    g = 9.80665    # m/s^2

    # Find layer for altitude h
    for i in range(len(layers) - 1):
        h_base, T_base, P_base, L = layers[i]
        h_next = layers[i+1][0]

        if h_base <= h < h_next:
            if abs(L) < 1e-10:
                # Isothermal layer
                T = T_base
                P = P_base * math.exp(-g * (h - h_base) / (R * T))
            else:
                # Gradient layer
                T = T_base + L * (h - h_base)
                P = P_base * (T / T_base) ** (-g / (L * R))

            rho = P / (R * T)
            return rho

    # If above model limit (86 km), return very low density
    return 0.0

def nearest_CNa_lookup(m):
    nearest = aero_df.index[np.argmin(np.abs(aero_df.index - m))]
    return aero_df.loc[nearest, "CNalpha (0 to 4 deg) (per rad)"]

def get_shear_force_and_bending_moment(N_n, w1, w2, x1):
    N_STATIONS = 100
    n_x1 = round(x1/ROCKET_LENGTH*N_STATIONS)
    shear_force_arr = np.zeros(N_STATIONS+1)
    bending_moment_arr = np.zeros(N_STATIONS+1)

    for i in range(0,n_x1):
        x = i/N_STATIONS * ROCKET_LENGTH
        shear_force_arr[i] = N_n - (w1 * x)
        bending_moment_arr[i] = (N_n * x) - (w1 * x**2 / 2)
    V1 = N_n - (w1 * x1)

    for i in range(n_x1,N_STATIONS):
        x = i/N_STATIONS * ROCKET_LENGTH
        shear_force_arr[i] = V1 - (w2*(x-x1))
        bending_moment_arr[i] = (V1 * x) + (w2 * ((x1 * x) + (1/2 * ROCKET_LENGTH**2) - (1/2 * x**2))) - (ROCKET_LENGTH * (V1 + (w2 * x1)))

    return [shear_force_arr, bending_moment_arr]

def get_max_bending_moment(N_n, w1, w2, x1):
    """
    Uses the Nakka formula for max bending moment
    This should be replaced
    """

    # Compute max_M_1
    try:
        max_M_1 = abs(N_n**2 / (2*w1))
    except FloatingPointError:
        max_M_1 = np.nan
    
    # Compute max_M_2
    try:
        V1 = N_n - (w1 * x1)
        max_M_2 = (V1**2 / (2*w2)) - (V1 * (ROCKET_LENGTH - x1)) + (w2 * ((0.5 * ROCKET_LENGTH**2) + (0.5 * x1**2) - (ROCKET_LENGTH * x1)))
    except FloatingPointError:
        max_M_2 = np.nan

    # Replace nan with 0
    if np.isnan(max_M_1):
        max_M_1 = 0.0
    if np.isnan(max_M_2):
        max_M_2 = 0.0

    max_M = max(max_M_1, max_M_2)
    return max_M

def calculate_CG(mass_arr):
    """
    Computes CG over time given the prop tank positions, O/F ratio, and starting/ending mass
    Assumes that tank fill percentage is equal at all times
    """
    dry_mass = mass_arr[-1]
    prop_mass = mass_arr - dry_mass
    fuel_mass = prop_mass / (1 + OF_RATIO)
    lox_mass = prop_mass * OF_RATIO / (1 + OF_RATIO)

    fill_fraction = np.clip(prop_mass / prop_mass[0], 0, 1)
    fuel_length = fill_fraction * FUEL_TANK_LENGTH
    lox_length = fill_fraction * LOX_TANK_LENGTH
    CG_fuel = FUEL_TANK_POSITION + (FUEL_TANK_LENGTH - fuel_length / 2)
    CG_lox  = LOX_TANK_POSITION  + (LOX_TANK_LENGTH  - lox_length  / 2)

    CG_arr = (CG_DRY*dry_mass + CG_fuel*fuel_mass + CG_lox*lox_mass) / mass_arr

    return CG_arr

"""
Array generation
"""

# raw dataframes
aero_df = pd.read_csv("CD Test.csv").set_index('Mach')
aero_df = aero_df[~aero_df.index.duplicated(keep='first')] # 0 deg AOA numbers
flight_df = pd.read_csv("Flight Test.csv")

# raw arrays
time_arr = flight_df["Time (sec)"].to_numpy()
vel_arr = flight_df['Velocity (ft/sec)'].to_numpy()/3.281
alt_arr = flight_df['Altitude (ft)'].to_numpy()/3.281
mach_arr = flight_df['Mach Number'].to_numpy()
mass_arr = flight_df['Weight (lb)'].to_numpy()/2.205 # kg
CP_arr = flight_df['CP (in)'].to_numpy()/39.37 # m

# derived arrays
rho_arr = np.zeros(len(alt_arr))
rho_arr = np.array([rho(h+LAUNCH_ALT) for h in alt_arr]) #kg/m^3
q_arr = np.zeros(len(alt_arr))
q_arr = 0.5*rho_arr*abs(vel_arr)**2 #Pa
CNalpha_arr = np.array([nearest_CNa_lookup(m) for m in mach_arr])
CNalpha_fin_arr = CNalpha_arr - 2

# prevent alpha from dividing by 0 and replace with pi/2
alpha_arr = np.zeros_like(vel_arr)
mask = vel_arr != 0
alpha_arr[mask] = 2 * np.arctan(WIND_GUST / abs(vel_arr[mask]))
alpha_arr[~mask] = np.pi/2

N_arr = q_arr * AREF * alpha_arr * CNalpha_arr # Multiplied by 2 to simulate wind gust going away while at opposite AOA during oscilitory response
N_fin_arr = q_arr * AREF * alpha_arr * CNalpha_fin_arr
N_nose_arr = q_arr * AREF * alpha_arr * 2
CG_arr = calculate_CG(mass_arr)
Stability_arr = (CP_arr - CG_arr) / ROCKET_LENGTH * 100

"""
# Stability plot
plt.figure(figsize=(10, 6))

plt.plot(time_arr[:np.argmax(alt_arr)+1], Stability_arr[:np.argmax(alt_arr)+1])

plt.xlabel("Time (sec)")
plt.ylabel("Stability (%)")
plt.title("Stability vs Time")

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()
"""

# Bending calculations
"""
The equations used here are from Nakka, they do not work with how high CNalpha is.
A negative force is needed on top of the CG to counteract the moment induced by the bottom.
This breaks all of the formulas, especially when the force required passes back through 0.
When the distributed load is 0, the bending moment goes to infinity
"""
x1_arr = CG_arr
x2_arr = ROCKET_LENGTH - CG_arr

w2_arr = (N_fin_arr * (2*x2_arr + x1_arr) - N_nose_arr*x1_arr) / (x2_arr**2 + x1_arr*x2_arr)
w1_arr = (N_nose_arr + N_fin_arr - (w2_arr*x2_arr))/x1_arr

max_M_arr = np.array([
    get_max_bending_moment(N_nose_arr[i], w1_arr[i], w2_arr[i], x1_arr[i])
    for i in range(len(CG_arr)) #len(CG_arr)
])

#debug
print(max_M_arr[0:100])

"""
Max load values
"""

# max velocity
max_vel = np.max(vel_arr)

print("Max velocity (m/s):", max_vel)

# max q
max_q = np.max(q_arr)
time_at_max_q = time_arr[np.argmax(q_arr)]

print("Max q (psi): ", max_q/6085)

# max N
max_N_fin = np.max(N_fin_arr)
max_N_nose = N_nose_arr[np.argmax(N_fin_arr)]
time_at_max_N_fin = time_arr[np.argmax(N_fin_arr)]

print("Max N_Fin (lbf): ", max_N_fin/4.448)
print("Max N_Nose (lbf): ", max_N_nose/4.448)

max_M = np.max(max_M_arr)
time_at_max_M = time_arr[np.argmax(max_M_arr)]
print(max_M_arr[0:100])

print("Max M (ft-lbf):", max_M/1.356)

"""
PREP DATA FOR PLOTTING
"""

# Cut off plots at burnout
burnout_idx = np.argmax(mass_arr <= mass_arr[-1])

time_plot   = time_arr[:burnout_idx+1]              # s
q_plot      = q_arr[:burnout_idx+1] / 6085          # psi
N_fin_plot  = N_fin_arr[:burnout_idx+1] / 4.448     # lbf
mach_plot   = mach_arr[:burnout_idx+1]
max_M_plot  = max_M_arr[:burnout_idx+1] / 1.356     # ft·lbf

# Event times
time_max_q     = time_arr[np.argmax(q_arr)]
time_max_N_fin = time_arr[np.argmax(N_fin_arr)]
time_max_M     = time_arr[np.nanargmax(max_M_arr)]

# Clip events to plotted window
time_max_q     = min(time_max_q, time_plot[-1])
time_max_N_fin = min(time_max_N_fin, time_plot[-1])
time_max_M     = min(time_max_M, time_plot[-1])

# Create figure with 4 plots
fig, axs = plt.subplots(
    4, 1,
    figsize=(11, 10),
    sharex=True,
    gridspec_kw={"hspace": 0.08}
)

# Dynamic Pressure Plot
axs[0].plot(time_plot, q_plot, linewidth=2)
axs[0].axvline(time_max_q, linestyle=":", linewidth=2)
axs[0].set_ylabel("q (psi)")
axs[0].grid(True)

# Fin Normal Force Plot
axs[1].plot(time_plot, N_fin_plot, linewidth=2)
axs[1].axvline(time_max_N_fin, linestyle=":", linewidth=2)
axs[1].set_ylabel("Fin Normal Force (lbf)")
axs[1].grid(True)

# Max Bending Moment Plot
axs[2].plot(
    time_plot,
    max_M_plot,
    linewidth=2
)
axs[2].axvline(time_max_M, linestyle=":", linewidth=2)
axs[2].set_ylabel("Max Bending Moment (ft·lbf)")
axs[2].set_xlabel("Time (s)")
axs[2].grid(True)
axs[2].set_ylim(0, 40000)

# Mach Number Plot
axs[3].plot(time_plot, mach_plot, linewidth=2, linestyle="--")
axs[3].set_ylabel("Mach")
axs[3].grid(True)

# Formatting
for ax in axs:
    ax.tick_params(width=1.5, length=6)

plt.suptitle(
    "Flight Loads vs Time (to Burnout)",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

"""
# SHEAR FORCE + BENDING MOMENT PLOT

V_arr, M_arr = get_shear_force_and_bending_moment(max_N_nose, w1_arr[valid_idx_M][np.argmax(max_M_arr[valid_idx_M])], w2_arr[valid_idx_M][np.argmax(max_M_arr[valid_idx_M])], x1_arr[valid_idx_M][np.argmax(max_M_arr[valid_idx_M])])
x = np.linspace(0, ROCKET_LENGTH*39.37, N_STATIONS + 1)

fig2, ax2 = plt.subplots()

# Left axis: bending moment (blue)
ax2.plot(x, M_arr/1.356, color='blue', label='Bending Moment')
ax2.set_xlabel('Position (in)')
ax2.set_ylabel('Bending Moment (ft·lbf)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.grid(True)

# Right axis: shear force (red)
ax_right = ax2.twinx()
ax_right.plot(x, V_arr/4.448, color='red', linestyle='--', label='Shear Force')
ax_right.set_ylabel('Shear Force (lbf)', color='red')
ax_right.tick_params(axis='y', labelcolor='red')

# Optional: combined legend
lines_l, labels_l = ax2.get_legend_handles_labels()
lines_r, labels_r = ax_right.get_legend_handles_labels()
ax2.legend(lines_l + lines_r, labels_l + labels_r)

plt.show()
"""
