import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

TIMESTEP_MS = 10  # milliseconds
TIMESTEP = TIMESTEP_MS / 1000  # seconds
MAX_TANK_SET = 590 * 6085 # Pa

# Temporary, replace with isp at throttle level using CEA
INITIAL_ISP = 240 # s

FEED_LOSS = 100 * 6085 # Pa
MIN_STIFFNESS = .15

with open("vehicleParameters.csv", newline="") as f:
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

GAMMA = 1.4
SPECIFIC_GAS_CONST = 287

g0 = 9.80665 # m/s^2

def get_rho(h):
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

import math

def get_temperature(h):
    """ 
    Compute air temperature (K) at altitude h (meters)
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

    # Find layer for altitude h
    for i in range(len(layers) - 1):
        h_base, T_base, _, L = layers[i]
        h_next = layers[i+1][0]

        if h_base <= h < h_next:
            if abs(L) < 1e-10:
                # Isothermal layer
                return T_base
            else:
                # Gradient layer
                return T_base + L * (h - h_base)

    # If above model limit (86 km), return top layer temperature
    return layers[-1][1]

def nearest_CNa_lookup(m, aero_df):
    nearest = aero_df.index[np.argmin(np.abs(aero_df.index - m))]
    return aero_df.loc[nearest, "CNalpha (0 to 4 deg) (per rad)"]

def get_gravity(h):
    R_earth = 6371000   # meters
    return g0 * (R_earth / (R_earth + h))**2

def get_Cd(m, aero_df, thrust):
    nearest = aero_df.index[np.argmin(np.abs(aero_df.index - m))]
    if thrust > 0:
        return aero_df.loc[nearest, "CD Power-On"]
    else:
        return aero_df.loc[nearest, "CD Power-Off"]
    
def rk4_step(y, dt, thrust, aero_df):
    h, v, m = y

    def derivatives(state):
        h_, v_, m_ = state
        rho = get_rho(h_ + LAUNCH_ALT)
        Cd = get_Cd(v_/((GAMMA*SPECIFIC_GAS_CONST*get_temperature(h_+LAUNCH_ALT))**0.5), aero_df, thrust)
        drag = 0.5 * rho * v_**2 * Cd * AREF
        gravity = m_ * 9.81
        a = (thrust - drag - gravity) / m_
        m_dot = thrust / (INITIAL_ISP * g0)
        if m_ <= dry_mass:
            return np.array([v_, 0.0, 0.0])
        return np.array([v_, a, -m_dot])

    k1 = derivatives(y)
    k2 = derivatives(y + 0.5 * dt * k1)
    k3 = derivatives(y + 0.5 * dt * k2)
    k4 = derivatives(y + dt * k3)

    y_next = y + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    # Enforce dry mass
    if y_next[2] < dry_mass:
        y_next[2] = dry_mass

    return y_next

# dataframes
aero_df = pd.read_csv("CD Test.csv").set_index('Mach')
aero_df = aero_df[~aero_df.index.duplicated(keep='first')] # 0 deg AOA numbers
flight_df = pd.read_csv("Flight Test.csv")
wet_mass = flight_df['Weight (lb)'].to_list()[0] / 2.205 # kg
dry_mass = flight_df['Weight (lb)'].to_list()[-1] / 2.205 # kg

min_thrust = int(wet_mass * 9.81 * 1.1) # N
max_thrust = int(wet_mass * 9.81 * 10) # N

"""
Flight simulation:
"""

best_apogee = (0,0)
thrust_values = np.linspace(min_thrust, max_thrust, 10, dtype=int)

for thrust in thrust_values:
    apogee = 0
    max_mach = 0
    max_vel = 0
    print("\nThrust: " + str(int(thrust/4.44822)) + " lbf")

    # Initialize arrays
    time_arr = [0]
    alt_arr = [0]
    velocity_arr = [0]
    thrust_arr = [thrust]
    drag_arr = [0]
    mass_arr = [wet_mass]
    mach_arr = [0]

    while not apogee:
        # Increment time
        time_arr.append(time_arr[-1] + TIMESTEP_MS)

        # Forces
        gravity = get_gravity(alt_arr[-1] + LAUNCH_ALT) * mass_arr[-1]  # N
        rho = get_rho(alt_arr[-1] + LAUNCH_ALT)  # kg/m^3
        Cd = get_Cd(mach_arr[-1], aero_df, thrust_arr[-1])
        dynamic_pressure = 0.5 * rho * velocity_arr[-1]**2  # Pa
        drag = dynamic_pressure * Cd * AREF
        drag_arr.append(drag)

        # Resultant force
        force = thrust_arr[-1] - drag - gravity

        # Kinematics
        acceleration = force / mass_arr[-1]
        velocity_arr.append(velocity_arr[-1] + acceleration * TIMESTEP)
        alt_arr.append(alt_arr[-1] + velocity_arr[-2] * TIMESTEP + 0.5 * acceleration * TIMESTEP**2)

        # Mach
        speed_sound = (GAMMA * SPECIFIC_GAS_CONST * get_temperature(alt_arr[-1] + LAUNCH_ALT))**0.5
        mach_arr.append(velocity_arr[-1] / speed_sound)

        # Track max values
        max_vel = max(max_vel, velocity_arr[-1])
        max_mach = max(max_mach, mach_arr[-1])

        # Mass Flow
        m_dot = thrust_arr[-1] / (INITIAL_ISP * g0)  # kg/s
        new_mass = mass_arr[-1] - m_dot * TIMESTEP
        if new_mass <= dry_mass:
            new_mass = dry_mass
            thrust_arr.append(0)
        else:
            thrust_arr.append(thrust)
        mass_arr.append(new_mass)

        # Apogee detection
        if alt_arr[-1] < alt_arr[-2]:
            apogee = alt_arr[-2]
            print(f"apogee: {apogee:.1f} m")
            print(f"Max Mach: {max_mach:.2f}")
            print(f"Max velocity: {max_vel:.1f} m/s")
            break

        # Sim Timeout
        if time_arr[-1] > 500*1000:
            print("Simulation timeout")
            break

    # Update best apogee
    if apogee > best_apogee[0]:
        best_apogee = (apogee, thrust)

# Summary
print("\nBest thrust: " + str(int(best_apogee[1]/4.44822)) + " lbf")
print("Initial TWR: " + str(best_apogee[1]/(9.81*wet_mass)))

# Forces plot for last run
# Change this to best run or maybe a few runs
gravity_arr = [m * 9.81 for m in mass_arr]
time_sec = np.array(time_arr) / 1000  # seconds

plt.figure(figsize=(10,6))
plt.plot(time_sec, thrust_arr, label='Thrust (N)', color='r')
plt.plot(time_sec, drag_arr, label='Drag (N)', color='b')
plt.plot(time_sec, gravity_arr, label='Gravity (N)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Rocket Forces vs Time (Last Thrust Run)')
plt.grid(True)
plt.legend()
plt.show()