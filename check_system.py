import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from amuse.lab import units, Particle
from amuse.units.optparse import OptionParser
from amuse.io import read_set_from_file, write_set_to_file

from planet_and_moons import get_planets_and_moons
from JPL import get_keplerian_elements, parse_vector_data

# Setup folders
plots_folder = "Plots/"
files_folder = "Files/"
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(files_folder, exist_ok=True)

# Matplotlib styles
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize': 12
})

def JPL_coordinates(solarsystem, date, name):
    """
    Function that generates a dataframe of cartesian coordinates straight from JPL at certain date
    Returns dataframe"""
    x, y, z, vx, vy, vz = [], [], [], [], [], []
    AU = (1 | units.AU).value_in(units.km)

    for particle in tqdm(solarsystem, desc="Querying JPL"):
        raw = get_keplerian_elements(particle.id, '@0', date, 'vector')
        #print(raw.text)
        parsed = parse_vector_data(raw, particle.name, date)
        x.append(parsed['X'] / AU)
        y.append(parsed['Y'] / AU)
        z.append(parsed['Z'] / AU)
        vx.append(parsed['VX'])
        vy.append(parsed['VY'])
        vz.append(parsed['VZ'])

    df = pd.DataFrame({
        "date": date,
        "object_id": solarsystem.id,
        "object_name": solarsystem.name,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz
    })

    filename = os.path.join(files_folder, f"JPL_coordinates_{name}.csv")
    df.to_csv(filename, index=False)
    return df

def histogram_differences(x_diff, y_diff, z_diff, vx_diff, vy_diff, vz_diff, name, save=False):
    """
    Function that plots the difference between JPL cartesian coordinates and JPL orbital elements
    that have been transformed to cartesian coordinates to check accuracy of conversion
    Saves two plots with histograms of position difference and velocity difference.
    """

    coords = np.concatenate([x_diff, y_diff, z_diff])
    vels = np.concatenate([vx_diff, vy_diff, vz_diff])
    bins_coords = np.linspace(coords.min(), coords.max(), 100)
    bins_vels = np.linspace(vels.min(), vels.max(), 100)

    plt.figure()
    plt.hist(x_diff, bins=bins_coords, alpha=0.3, label='Δx', color='blue')
    plt.hist(y_diff, bins=bins_coords, alpha=0.3, label='Δy', color='orange')
    plt.hist(z_diff, bins=bins_coords, alpha=0.3, label='Δz', color='green')
    plt.xlabel('Δr [AU]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.title("Difference between Cartesian coordinates directly from JPL \n and derived from orbital elements in JPL")
    if save:
        plt.savefig(os.path.join(plots_folder, f'Coordinate_differences_JPL_Amuse_{name}.pdf'), dpi=450, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist(vx_diff, bins=bins_vels, alpha=0.3, label='Δvx', color='blue')
    plt.hist(vy_diff, bins=bins_vels, alpha=0.3, label='Δvy', color='orange')
    plt.hist(vz_diff, bins=bins_vels, alpha=0.3, label='Δvz', color='green')
    plt.xlabel('Δv [km/s]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.xlim()
    plt.title("Difference between Cartesian velocities directly from JPL \n and derived from orbital elements in JPL")
    if save:
        plt.savefig(os.path.join(plots_folder, f'Velocity_differences_JPL_Amuse_{name}.pdf'), dpi=450, bbox_inches='tight')
    plt.close()

def calculate_difference_perc(system, jpl_df, name, save=False):
    x_diff = [(system.x[i].value_in(units.AU) - jpl_df['x'][i]) / system.x[i].value_in(units.AU) for i in range(len(system))]
    y_diff = [(system.y[i].value_in(units.AU) - jpl_df['y'][i]) / system.y[i].value_in(units.AU) for i in range(len(system))]
    z_diff = [(system.z[i].value_in(units.AU) - jpl_df['z'][i]) / system.z[i].value_in(units.AU) for i in range(len(system))]
    vx_diff = [(system.vx[i].value_in(units.kms) - jpl_df['vx'][i]) / system.vx[i].value_in(units.kms) for i in range(len(system))]
    vy_diff = [(system.vy[i].value_in(units.kms) - jpl_df['vy'][i]) / system.vy[i].value_in(units.kms) for i in range(len(system))]
    vz_diff = [(system.vz[i].value_in(units.kms) - jpl_df['vz'][i]) / system.vz[i].value_in(units.kms) for i in range(len(system))]

    histogram_differences(x_diff, y_diff, z_diff, vx_diff, vy_diff, vz_diff, name, save=save)

    print(f"\nDifference summary for {name}:")
    for label, diff in zip(['Δx', 'Δy', 'Δz', 'Δvx', 'Δvy', 'Δvz'],
                           [x_diff, y_diff, z_diff, vx_diff, vy_diff, vz_diff]):
        print(f"{label}: min={np.min(diff):.2e}, max={np.max(diff):.2e}, mean={np.mean(diff):.2e}")

def calculate_difference(system, jpl_df, name, save=False):
    """
    Function to calculate differences between JPL cartesian coordinates and derived cartesian coordinates from JPL orbital elements
    Also saves all differences to a csv in \Files
    """
    x_diff = [system.x[i].value_in(units.AU) - jpl_df['x'][i] for i in range(len(system))]
    y_diff = [system.y[i].value_in(units.AU) - jpl_df['y'][i] for i in range(len(system))]
    z_diff = [system.z[i].value_in(units.AU) - jpl_df['z'][i] for i in range(len(system))]
    vx_diff = [system.vx[i].value_in(units.kms) - jpl_df['vx'][i] for i in range(len(system))]
    vy_diff = [system.vy[i].value_in(units.kms) - jpl_df['vy'][i] for i in range(len(system))]
    vz_diff = [system.vz[i].value_in(units.kms) - jpl_df['vz'][i] for i in range(len(system))]

    histogram_differences(x_diff, y_diff, z_diff, vx_diff, vy_diff, vz_diff, name, save=save)

    print(f"\nAbsolute difference summary for {name}:")
    print(f"Units: AU for position, km/s for velocity\n")
    for label, diff in zip(['Δx', 'Δy', 'Δz', 'Δvx', 'Δvy', 'Δvz'],
                           [x_diff, y_diff, z_diff, vx_diff, vy_diff, vz_diff]):
        print(f"{label}: min={np.min(diff):.4e}, max={np.max(diff):.4e}, mean={np.mean(diff):.4e}")

    diff_df = pd.DataFrame({
        "object_name": system.name,
        "Δx (AU)": x_diff,
        "Δy (AU)": y_diff,
        "Δz (AU)": z_diff,
        "Δvx (km/s)": vx_diff,
        "Δvy (km/s)": vy_diff,
        "Δvz (km/s)": vz_diff
    })

    diff_df.to_csv(os.path.join(files_folder, f"Differences_absolute_{name}.csv"), index=False)

def check_planets_and_moons(solarsystem, date, name="check", save=True, fraction=0.25):
    """
    Select 25% of planets and moons object to compare JPL cartesian coordinates to AMUSE generated cartesian coordinated
    Saves two histograms with velocity and position differences in Plots
    """
    system = solarsystem.select(lambda t: t in ["Planet", "Moon"], ["type"])
    
    N = int(len(system) * fraction)
    subset = system[random.sample(range(len(system)), N)]
    #subset = system(random.sample(system, N))

    jpl_df = JPL_coordinates(subset, date, name)
    calculate_difference(subset, jpl_df, name, save=save)

def new_option_parser():
    parser = OptionParser()
    parser.add_option("--save", action="store_true", dest="save", default=True)
    parser.add_option("--show", action="store_true", dest="show", default=False)
    parser.add_option("--date", dest="current_date", default="01-01-2024")

    return parser.parse_args()

if __name__ == "__main__":
    options, args = new_option_parser()
    filename = os.path.join("Solarsystem.hdf5")

    if os.path.exists("Solarsystem.hdf5"):
        loaded = read_set_from_file("Solarsystem.hdf5", format="hdf5")
        check_planets_and_moons(loaded, options.current_date)
