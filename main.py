import os
import matplotlib.pyplot as plt
import numpy as np

from amuse.units.optparse import OptionParser
from amuse.io import write_set_to_file
from amuse.lab import units, constants, nbody_system
from amuse.lab import Particle, Particles

from planet_and_moons import get_planets_and_moons
from sbdb import get_particleset
from check_system import check_planets_and_moons

plots_folder = "Plots/"
files_folder = "Files/"
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(files_folder, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize': 12
})

def plot_full_system(solarsystem, lim=40,perc=0.25):
    """
    Function to plot the entire generated solarsystem, within a certain limit (default = 40 AU)
    Asteroids and comets are only plotted a certain percentage, due to large abundance (default = 25%)
    Returns plot of system in folder "Plots"
    """
    planets = solarsystem[solarsystem.type == 'Planet']
    moons = solarsystem[solarsystem.type == "Moon"]
    asteroids_unn = solarsystem[solarsystem.type == "asteroid unnumbered"]
    asteroids_nr = solarsystem[solarsystem.type == "asteroid numbered"]
    comets_unn = solarsystem[solarsystem.type == "comet unnumbered"]
    comets_nr = solarsystem[solarsystem.type == "comet numbered"]

    plt.figure()
    theta = np.linspace(0, 2 * np.pi, 100)
    radii = np.linspace(0.00, lim, 10)
    for r in radii:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.plot(x, y, color='grey', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.scatter(solarsystem[0].x.value_in(units.AU), solarsystem[0].y.value_in(units.AU), color='gold', label='Sun', zorder=10, s=25)
    plt.scatter(planets.x.value_in(units.AU), planets.y.value_in(units.AU), color='tab:red', label='Planets', zorder=8, s=15)
    plt.scatter(moons.x.value_in(units.AU), moons.y.value_in(units.AU), color='tab:orange', label='Moons', zorder=7, s=25)

    def plot_random_subset(data, color, perc=0.25,label=None):
        if len(data) == 0:
            return
        subset_size = int(len(data) * perc)
        if subset_size == 0:
            return
        random_indices = np.random.choice(len(data), subset_size, replace=False)
        plt.scatter(data.x.value_in(units.AU)[random_indices],
                    data.y.value_in(units.AU)[random_indices],
                    color=color, label=label, s=0.5, zorder=0, alpha=0.75)

    plot_random_subset(asteroids_nr, 'tab:green', perc,f'{int(perc*100)}% Asteroids')
    plot_random_subset(asteroids_unn, 'tab:green')
    plot_random_subset(comets_nr, 'tab:blue', perc,f'{int(perc*100)}% Comets')
    plot_random_subset(comets_unn, 'tab:blue')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.savefig(f"Plots/Solarsystem_realisation_{lim}AU.pdf", bbox_inches='tight')
    plt.close()

def create_solar_system(date, planets=True, moons=True, asteroids_nr=True, asteroids_unn=True, comets_nr=True, comets_unn=True, limit=100):
    """
    Function used to create a solar system AMUSE particle set based on JPL orbital elements, with options:
    if options.add_planets: planets are added
    if options.add_moons: moons are added
    if options.add_asteroids_nr: asteroids numbered are added
    if options.add_asteroids_unn: asteroids unnumbered are added
    if options.add_comets_nr: numbered comets are added
    if options.add_comets_unn: unnumbered comets are added
    
    Solar system is moved to center
    """

    #### Create basic system with sun at pos 0, velocity 0
    solarsystem = Particles()

    sun = Particle()
    sun.name = 'Sun'
    sun.mass = 1 | units.MSun  
    sun.radius = 1 | units.km  
    sun.position = [0, 0, 0] | units.AU  
    sun.velocity = [0, 0, 0] | units.kms 
    sun.id=10
    sun.type="Star"

    solarsystem.add_particle(sun)

    if planets:
        print("Adding planets (from JPL)")
        planets=get_planets_and_moons(date,planets=True,moons=False)
        solarsystem.add_particles(planets)

    if moons:
        print("Adding moons (from JPL)")
        moons=get_planets_and_moons(date,planets=False,moons=True)
        solarsystem.add_particles(moons)

    if asteroids_nr:
        print("Adding numbered asteroids (from SBDB)")
        ast_nr = get_particleset("asteroid", "numbered", date=date, limit=limit)
        solarsystem.add_particles(ast_nr)

    if asteroids_unn:
        print("Adding unnumbered asteroids (from SBDB)")
        ast_unn = get_particleset("asteroid", "unnumbered", date=date, limit=limit)
        solarsystem.add_particles(ast_unn)

    if comets_nr:
        print("Adding numbered comets (from SBDB)")
        com_nr = get_particleset("comet", "numbered", date=date, limit=limit)
        solarsystem.add_particles(com_nr)

    if comets_unn:
        print("Adding unnumbered comets (from SBDB)")
        com_unn = get_particleset("comet", "unnumbered", date=date, limit=limit)
        solarsystem.add_particles(com_unn)

    print("Moving the system to center")
    solarsystem.move_to_center()
    
    return solarsystem

def new_option_parser():
    result = OptionParser()
    result.add_option("--date", dest="current_date", default="01-01-2024")
    result.add_option("--limit", dest="limit", type="int", default=250, help="Limit per small-body group")


    result.add_option("--add_planets", dest="add_planets", action="store_true", default=True)
    result.add_option("--add_moons", dest="add_moons", action="store_true", default=True)

    result.add_option("--add_asteroids_nr", dest="add_asteroids_nr", action="store_true", default=True)
    result.add_option("--add_asteroids_unn", dest="add_asteroids_unn", action="store_true", default=True)

    result.add_option("--add_comets_nr", dest="add_comets_nr", action="store_true", default=True)
    result.add_option("--add_comets_unn", dest="add_comets_unn", action="store_true", default=True)

    return result

if __name__ in ('__main__', '__plot__'):
    parser = new_option_parser()
    options, args = parser.parse_args()

    solarsystem = create_solar_system(
        date=options.current_date,
        planets=options.add_planets,
        moons=options.add_moons,
        asteroids_nr=options.add_asteroids_nr,
        asteroids_unn=options.add_asteroids_unn,
        comets_nr=options.add_comets_nr,
        comets_unn=options.add_comets_unn,
        limit=options.limit
    )

    print(solarsystem)
    filename = "Solarsystem.hdf5"
    write_set_to_file(solarsystem, filename, format='hdf5', overwrite_file=True)

    plot_full_system(solarsystem)
    #Function that checks the derived cartesian coordinates of planets+moons with JPL cartesian coordinates
    check_planets_and_moons(solarsystem, date=options.current_date)
