import numpy as np
import pandas as pd
from tqdm import tqdm

from amuse.lab import units, Particle, Particles
from amuse.lab import nbody_system
from amuse.community.kepler.interface import Kepler
from amuse.io import write_set_to_file

from JPL import get_keplerian_elements, get_orbital_elements_from_JPL, major_bodies_list

massless_moons_name = []
massless_moons_ID = []

def new_kepler():
    converter = nbody_system.nbody_to_si(1 | units.MSun, 1 | units.au)
    k = Kepler(converter, redirection='none')
    k.initialize_code()
    return k

def get_position_of_secondary_particle(mass_sun, mass_planet, semi, ecc, incl, argument, longitude, mean_anomaly, delta_t=0.|units.day):
    """
    Function to get cartesian coordinates using orbital elements
    Used for planets wrt Sun and for moons wrt planets.
    Returns cartesian coordinates

    Code copied from: amuse/src/amuse/icc/solar_system_moons.py
    """
    k = new_kepler()
    k.initialize_from_elements(mass=(mass_sun + mass_planet), semi=semi, ecc=ecc, mean_anomaly=mean_anomaly)
    k.transform_to_time(time=delta_t)
    r = k.get_separation_vector()
    v = k.get_velocity_vector()
    k.stop()

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

def get_system(data, central_name, central_mass, central_radius, central_id, central_type, object_type):
    """
    Function to create the (sub)system, by inputting data of the central object and a dataframe of orbital elements and characteristics of 
    accompanying objects (planets wrt sun, or moons wrt planet)
    Function returns particleset with central object and surrouning objects
    """
    parts = Particles(len(data))
    parts.mass = np.array(data['mass']) / (1.989e30) | units.MSun
    parts.radius = np.array(data['radius']) | units.km
    parts.name = np.array(data['name'], dtype=str)
    parts.id = data['ID'].values
    parts.parent_id = central_name
    parts.type = object_type

    ecc = np.array(data['ecc'])
    semi = np.array(data['semi']) | units.km
    mean_anomaly = np.deg2rad(data['mean_anomaly'])
    incl = np.deg2rad(data['incl'])
    lon = np.deg2rad(data['longitude'])
    arg = np.deg2rad(data['argument'])

    center = Particle()
    center.name = central_name
    center.mass = central_mass | units.MSun
    center.radius = central_radius | units.km
    center.position = [0, 0, 0] | units.AU
    center.velocity = [0, 0, 0] | units.kms
    center.id = central_id
    center.type = central_type

    for i in range(len(parts)):
        r, v = get_position_of_secondary_particle(center.mass, parts[i].mass, semi[i], ecc[i], incl[i], arg[i], lon[i], mean_anomaly[i])
        parts[i].position = r
        parts[i].velocity = v

    return center, parts

def planet_moon_system(major_bodies, planet_id, planet_name, date):
    """
    Creating subsystem of planet with moons for planet of ID=planet_id
    Returns full subsystem as AMUSE particleset
    """
    moons_id,moons_name=[],[]

    prefix = int(planet_id) // 100
    short_min = prefix * 100
    short_max = short_min + 100
    long_min = prefix * 10000
    long_max = long_min + 10000

    for i in range(len(major_bodies)):
        body_id = int(major_bodies["ID"][i])
        if ((short_min <= body_id < short_max) or (long_min <= body_id < long_max)) and body_id != planet_id:
            moons_id.append(major_bodies["ID"][i])
            moons_name.append(major_bodies["Name"][i])

    planet_el = get_orbital_elements_from_JPL(planet_id, planet_name, date, '@sun')
    planet_mass = planet_el['mass'] / (1 | units.MSun).value_in(units.kg)
    planet_radius = planet_el['radius']

    moons_el = []
    print(f"Adding moons of {planet_name}")
    for i, moon in tqdm(enumerate(moons_id), total=len(moons_id)):
        if int(moon) == 635:
            continue
        moon_data = get_orbital_elements_from_JPL(moon, moons_name[i], date, f'@{planet_id}')
        moons_el.append(moon_data)

    moons_el = pd.DataFrame(moons_el)
    min_mass = moons_el['mass'].min()
    
    #If we want to find out what moon masses are approximated / not directly available in JPL
    for i in range(len(moons_el)):
        if pd.isna(moons_el['mass'][i]) or moons_el['mass'][i] <= 0:
            massless_moons_name.append(moons_name[i])
            massless_moons_ID.append(moons_id[i])
    #Assign mass of lowest mass moon to all moons without mass estimate in JPL
    moons_el.loc[(moons_el['mass'] <= 0) | (moons_el['mass'].isna()), 'mass'] = min_mass

    _, moons = get_system(moons_el, planet_name, planet_mass, planet_radius, planet_id, "Planet", "Moon")
    return moons

def get_planets_and_moons(date, planets=True,moons=True):
    """
    Function to create system of either all solarsystem planets (incl pluto), all moons, or both
    Generate data (IDs) for moons from JPL major_bodies list
    Creating particlesets for planets, and moons, and combining these to return full particleset
    """
    major_bodies = get_keplerian_elements("MB", "@sun", date, 'elements')
    major_bodies = major_bodies_list(major_bodies)
    ids = [199, 299, 399, 499, 599, 699, 799, 899, 999]
    names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]

    system = Particles()

    planet_data = []
    #print("Adding planets")
    for i, pid in tqdm(enumerate(ids), total=len(ids)):
        data = get_orbital_elements_from_JPL(pid, names[i], date, '@sun')
        planet_data.append(data)
    planet_df = pd.DataFrame(planet_data)
    _, planets_particles = get_system(planet_df, 'Sun', 1, 1, 10, "Star", "Planet")
    
    #Only add planets if planets=True
    if planets:
        system.add_particles(planets_particles)

    #Only add moons if moons=True
    if moons:
        #print("Adding moons")
        for planet in planets_particles.copy():  # avoid modifying while iterating
            if planet.name in ["Mercury", "Venus"]:
                continue
            moon_particles = planet_moon_system(major_bodies, planet.id, planet.name, date)
            moon_particles.position += planet.position
            moon_particles.velocity += planet.velocity
            system.add_particles(moon_particles)

    return system


if __name__ == '__main__':
    from amuse.units.optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--date", dest="date", default="01-01-2024")
    parser.add_option("--planets", action="store_true", dest="planets", default=True)
    parser.add_option("--moons", action="store_true", dest="moons", default=True)
    options, args = parser.parse_args()

    system = get_planets_and_moons(options.date, planets=options.planets,moons=options.moons)
    print(system)

    filename="Files/Planets_and_moons.hdf5"
    write_set_to_file(system,filename,format='hdf5',append_to_file=False)
