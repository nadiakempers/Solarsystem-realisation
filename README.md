# Solar System Realisation & Validation

This repository contains code to generate a realistic solar system particle set using data from NASA/JPL, simulate it using the AMUSE framework, and compare positions and velocities against JPL's own ephemerides.

## 🚀 Features

- Download asteroid and comet orbital elements from the JPL SBDB API
- Add planets and moons based on JPL Horizons data
- Estimate masses for small bodies based on albedo and diameter
- Compute Cartesian state vectors using Keplerian orbits
- Save full particle sets to HDF5
- Plot 2D system layout
- Compare generated state vectors to JPL reference values (position/velocity difference histograms)

## 🧠 Structure

- `main.py`: main script to build and visualise the system
- `sbdb.py`: handles SBDB queries and small-body particle creation
- `planet_and_moons.py`: fetches planets and moons from JPL
- `check_system.py`: validates generated objects against JPL data
- `JPL.py`: low-level JPL Horizons querying and data parsing
- `Plots/` and `Files/`: output folders for plots and data

## 📦 Requirements

- Python 3.8+
- [AMUSE framework](https://amusecode.github.io/)
- Required packages: `numpy`, `pandas`, `matplotlib`, `tqdm`, `requests`

## 🔧 Usage

Run from `main.py`: by customizing the command-line options

Run from terminal: by using the command-line inputs

- `--date`: Sets the target date for all orbital elements (format: `DD-MM-YYYY` depending on implementation).
- `--limit`: Number of small bodies (asteroids/comets) to include per category.
- `--add_planets`: Include all 8 planets incl pluto in the solar system.
- `--add_moons`: Include all moons (incl moons of pluto).
- `--add_asteroids_nr`: Include **numbered asteroids** (with known orbits).
- `--add_asteroids_unn`: Include **unnumbered asteroids** (less certain orbits).
- `--add_comets_nr`: Include **numbered comets**.
- `--add_comets_unn`: Include **unnumbered comets**.

```bash
python main.py --date 2024-01-01 --limit 100 --add_planets --add_moons --add_asteroids_nr --add_asteroids_unn --add_comets_nr --add_comets_unn
```

## 📤 Output

When the system is generated, several outputs are saved automatically:

- `Solarsystem.hdf5`:  
  The full AMUSE `Particles` set containing all solar system bodies and their properties (mass, position, velocity, etc.) at a certain date/epoch

- `Plots/Solarsystem_realisation_XAU.pdf`:  
  A 2D overview plot of the entire system, zoomed to the specified AU limit.

- `Plots/Coordinate_differences_JPL_Amuse_*.pdf` and `Plots/Velocity_differences_JPL_Amuse_*.pdf`:  
  Histograms showing the difference (of certain objects) between your calculated orbits and JPL reference data.

- `Files/*.csv`:  
  - Raw SBDB data downloaded from NASA in CSV format of small body  
  - JPL reference vectors (Cartesian position/velocity data per object)

You can use these outputs for analysis, debugging, or building further simulations.

## 📬 Contact

For questions, bug reports, or collaboration ideas, feel free to contact me via email or open an issue on this repository.

**Author:** Nadia Kempers 
**Email:** n.w.e.kempers@umail.leidenuniv.nl# Solarsystem-realisation
