import csv
import pandas as pd

"""
Pulls values required to run bendingLoads.py from Starting Point Excel
"""

def find_value(df, label, direction="right"):
    result = df.isin([label])
    coords = list(zip(*result.to_numpy().nonzero()))
    if not coords:
        return None  # label not found
    row, col = coords[0]
    if direction == "right":
        return df.iloc[row, col + 1]
    elif direction == "below":
        return df.iloc[row + 1, col]
    else:
        raise ValueError("direction must be 'right' or 'below'")

"""
Pull values
"""
EXCEL_PATH = "../../Starting Point.xlsx"

xls = pd.ExcelFile(EXCEL_PATH)
dV_df = pd.read_excel(xls, sheet_name="Delta V and Tank Sizing")
mass_df = pd.read_excel(xls, sheet_name="Mass Estimation")

# Pull the mass table out of mass_df
mass_table_df = mass_df.copy()
mass_table_df.columns = mass_table_df.iloc[0]
mass_table_df = mass_table_df[1:]
mass_table_df.reset_index(drop=True, inplace=True)
blank_rows = mass_table_df.isnull().all(axis=1)
if blank_rows.any():
    first_blank_index = blank_rows.idxmax()  # first blank row
    mass_table_df = mass_table_df.iloc[:first_blank_index]
mass_table_df.reset_index(drop=True, inplace=True)

# Set up parameter dictionary
parameter_dict = {}

# Map dv sheet python variables to their excel cell names
dv_param_map = {
    "AREF": "Barrel Outside C/S Area",
    "OF_RATIO": "O/F"
}

for py_var, excel_label in dv_param_map.items():
    value = find_value(dV_df, excel_label)
    parameter_dict[py_var] = value

# Convert units
parameter_dict["AREF"] = parameter_dict["AREF"]/1550 # in^2 to m^2

# Map mass table python variables to their excel row and column names
mass_map = {
    "ROCKET_LENGTH": ("TOTAL", "Length (in)"),
    "CG_DRY": ("TOTAL", "Arm (in)"),
    "CG_WET": ("WET", "Arm (in)"),
    "FUEL_TANK_LENGTH": ("Fuel Tank", "Length (in)"),
    "FUEL_TANK_POSITION": ("Fuel Tank", "Front Location (in)"),
    "LOX_TANK_LENGTH": ("LOX Tank", "Length (in)"),
    "LOX_TANK_POSITION": ("LOX Tank", "Front Location (in)")
}

for py_var, (item_name, col_name) in mass_map.items():
    row = mass_table_df.loc[mass_table_df["Item"] == item_name]
    if not row.empty:
        parameter_dict[py_var] = (row.iloc[0][col_name])/39.37 # inches to meters
    else:
        parameter_dict[py_var] = None

# View extracted values
for k, v in parameter_dict.items():
    print(f"{k}: {v}")

# Write dictionary to csv
with open('vegaParameters.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=parameter_dict.keys())
    writer.writeheader()
    writer.writerow(parameter_dict)