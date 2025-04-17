import numpy as np
from datetime import datetime, timedelta, date
import requests
import pandas as pd
from tqdm import tqdm
import os

from amuse.lab import units, constants, nbody_system
from amuse.lab import Particles, write_set_to_file
from amuse.community.kepler.interface import Kepler


def download_sbdb_csv(object_kind, numbered_state, limit, output_file):
    """
    Function to download csv file of SBDB objects for object_kind, numbered_state
    Saves csv file in folder Files
    """

    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    kind_code = "a" if object_kind == "asteroid" else "c"
    numbered_code = "n" if numbered_state == "numbered" else "u"

    params = {
        "sb-kind": kind_code,
        "sb-ns": numbered_code,
        "sb-xfrag": "1",
        "fields": "spkid,GM,full_name,albedo,diameter,name,epoch,e,a,i,om,w,ma",
        "limit": limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'fields' in data and 'data' in data:
        df = pd.DataFrame(data['data'], columns=data['fields'])
        df.to_csv(output_file, index=False)
        #print(f"{object_kind.title()} ({numbered_state}) saved to {output_file}")
    else:
        print(f"Failed to download {object_kind} ({numbered_state})")



def new_kepler():
    converter = nbody_system.nbody_to_si(1 | units.MSun, 1 | units.au)
    kepler = Kepler(converter, redirection='none')
    kepler.initialize_code()
    return kepler


def get_position_of_secondary_particle(mass_object, semi, ecc, incl, argument, longitude, mean_anomaly, epoch, epoch_target):
    """
    Function to get cartesian coordinates using orbital elements
    Used for comets and asteroids orbiting the Sun
    Returns cartesian coordinates

    Code copied from: amuse/src/amuse/icc/solar_system_moons.py
    """
    kepler = new_kepler()
    mass_sun = 1 | units.MSun
    mass_object = mass_object / (1.989 * 10**30) | units.MSun

    kepler.initialize_from_elements(mass=(mass_sun + mass_object),
                                    semi=semi,
                                    ecc=ecc,
                                    mean_anomaly=mean_anomaly)

    delta_t = (epoch_target - epoch) | units.day
    kepler.transform_to_time(time=delta_t)
    r = kepler.get_separation_vector()
    v = kepler.get_velocity_vector()
    kepler.stop()

    a1 = ([np.cos(longitude), -np.sin(longitude), 0.0],
          [np.sin(longitude), np.cos(longitude), 0.0],
          [0.0, 0.0, 1.0])
    a2 = ([1.0, 0.0, 0.0],
          [0.0, np.cos(incl), -np.sin(incl)],
          [0.0, np.sin(incl), np.cos(incl)])
    a3 = ([np.cos(argument), -np.sin(argument), 0.0],
          [np.sin(argument), np.cos(argument), 0.0],
          [0.0, 0.0, 1.0])
    A = np.dot(np.dot(a1, a2), a3)
    r_vec = np.dot(A, np.reshape(r, 3, 'F'))
    v_vec = np.dot(A, np.reshape(v, 3, 'F'))

    r[0], r[1], r[2] = r_vec
    v[0], v[1], v[2] = v_vec

    return r, v


def estimate_mass_asteroids(A, d):
    """
    Function to estimate mass of asteroids based on Albedo and diameter
    Use albedo to assign one of two densities (based on literature estimates), and use diameter to calculate mass
    Returns mass
    """
    r = d / 2
    if A <= 0.1:
        rho = 1.38 #g/cm^3
    else:
        rho = 2.71 #g/cm^3
    M = 10**12 * (4/3) * np.pi * (r**3) * rho
    return M

def estimate_mass_comets(d):
    """
    Function to estimate mass of comets based on diameter
    Assume average density of 0.5 g/cm^3 based on literature
    Returns mass
    """
    rho= 0.5 #g/cm^3
    r=d/2

    M=10**12*(4/3)*np.pi*(r**3)*rho
    return M

def create_particles(input_file, epoch_target_jd,object_kind,numbered_state):
    """
    Function that creates entire particle set of certain object_kind, numbered_state using SBDB info
    Reads created csv file and returns particleset
    """
    data = pd.read_csv(input_file)
    particles = Particles(len(data))

    G = constants.G.value_in((units.km**3 / (units.kg * units.s**2)))
    masses=[]

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        spkid = row['spkid']
        full_name = row['full_name']
        epoch = row['epoch']
        GM = row['GM']
        A = row['albedo']
        d = row['diameter']

        if GM > 0:
            M = GM / G
        else:
            if object_kind == "asteroid":
                M = estimate_mass_asteroids(A, d) if A > 0 else 0
            elif object_kind == "comet":
                M = estimate_mass_comets(d) if d > 0 else 0
            else:
                M = 0

        e = row['e']
        a = row['a'] | units.AU
        i = np.deg2rad(row['i'])
        om = np.deg2rad(row['om'])
        w = np.deg2rad(row['w'])
        ma = np.deg2rad(row['ma'])

        r, v = get_position_of_secondary_particle(M, a, e, i, w, om, ma, epoch, epoch_target_jd)

        particles[idx].position = r
        particles[idx].velocity = v
        particles[idx].mass = M | units.kg
        particles[idx].name = full_name
        particles[idx].radius = d / 2 | units.km if d>0 else 0 | units.km
        particles[idx].albedo = A if A>0 else 0
        particles[idx].id = spkid
        particles[idx].type=f"{object_kind} {numbered_state}"

        masses.append(M)
    
    #Estimate mass for all 0 mass particles based on other particles
    nonzero_masses = [m for m in masses if m > 0]
    if nonzero_masses:
        min_mass = min(nonzero_masses)
    else:
        min_mass = 0  #If no mass estimates, set M=0

    for p in particles:
        if p.mass.value_in(units.kg) == 0:
            p.mass = min_mass | units.kg

    print(particles)
    return particles

def remove_nan_particles(particles):
    """
    Remove objects with no valid orbital elements, and thus no valid cartesian coordinates
    """
    mask = (
        np.isnan(particles.x.value_in(units.m)) | np.isnan(particles.y.value_in(units.m)) | np.isnan(particles.z.value_in(units.m)) |
        np.isnan(particles.vx.value_in(units.m / units.s)) | np.isnan(particles.vy.value_in(units.m / units.s)) | np.isnan(particles.vz.value_in(units.m / units.s))
    )
    #print(f"Number of objects with NaN in position or velocity: {np.sum(mask)} of total {len(particles)} particles")
    return particles[~mask]

def get_particleset(object_kind, numbered_state, date, limit, save_hdf5=False):
    """
    Function that inputs object_kind, numbered_state, date and limit (number of objects) 
    Returns particleset, and if save=True, also saves file to Files folder
    """
    dt = datetime.strptime(date, "%d-%m-%Y")
    jd = 367 * dt.year - ((7 * (dt.year + ((dt.month + 9) // 12))) // 4) + ((275 * dt.month) // 9) + dt.day + 1721013.5

    base = f"{object_kind}_{numbered_state}"
    csv_file = f"Files/{base}.csv"
    hdf5_file = f"Files/{base}.hdf5"

    if not os.path.exists(csv_file):
        download_sbdb_csv(object_kind, numbered_state, limit, output_file=csv_file)

    particleset = create_particles(csv_file, jd,object_kind,numbered_state)
    particleset = remove_nan_particles(particleset) #remove all nan objects
    

    if save_hdf5:
        write_set_to_file(particleset, hdf5_file, format="hdf5", append_to_file=False)

    return particleset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sbclass", type=str, default="asteroid")
    parser.add_argument("--numbered", type=str, default="numbered")
    parser.add_argument("--limit", type=str, default="100")
    parser.add_argument("--date", type=str, default="01-01-2024")
    args = parser.parse_args()

    if not os.path.exists("Files/"):
        os.makedirs("Files/")
    if not os.path.exists("Plots/"):
        os.makedirs("Plots/")

    particles = get_particleset(args.sbclass, args.numbered, date=args.date, limit=args.limit, save_hdf5=True)
