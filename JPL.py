#File with helping codes to generate JPL data for planets & moons
#All functions are used in planet_and_moons.py

import re
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

### Function that generate JPL data ###

def get_keplerian_elements(target_id, center, day, ephem_type):
    """
    Function that uses API to return raw data of orbital elements 
    of a target at a certain day wrt to a certain center.
    """

    url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    ref_date = datetime.strptime(day, "%d-%m-%Y")
    stop_date = ref_date + timedelta(days=1) #needs a start and stop date

    params = {
        'format': 'text',
        'command': f"'{target_id}'",
        'center': center,
        'start_time': f"'{ref_date}'",
        'stop_time': f"'{stop_date}'",
        'step_size': '1d',
        'obj_data': 'YES',
        'make_ephem': 'YES',
        'ephem_type': ephem_type,
        'csv_format': 'yes'
    }

    return requests.get(url, params=params)

def major_bodies_list(response):
    """
    Function that generates the list of major bodies containing planets and moons
    ID, in order to generate orbital elements
    """
    text = response.text
    lines = text.strip().split('\n')[8:-2]

    rows = []
    for line in lines:
        parts = list(filter(None, line.split('  ')))
        if len(parts) >= 2:
            rows.append({'ID': parts[0], 'Name': ''.join(parts[1])})

    return pd.DataFrame(rows)[['ID', 'Name']]

### Functions that generate the mass and radius from raw data ###
def generate_mass(raw_data):
    """
    Function to generate the mass from the raw data
    Needs several mass patterns, as JPL does not use one clear pattern
    """
    mass_patterns = [
        r"Mass x10\^(\d+) \(kg\)\s*=\s*([\d.]+)",
        r"Mass x 10\^(\d+) \(g\)\s*=\s*([\d.]+)",
        r"Mass, x10\^(\d+) \(kg\)\s*=~\s*([\d.]+)",
        r"Mass, x10\^(\d+) kg\s*=\s*([\d.]+)",
        r"Mass, x10\^(\d+) kg\s*=\s*([\d.]+)",
        r"Mass \(10\^(\d+) kg \)\s*=\s*([\d.]+)",
    ]

    gm_pattern = [r"GM \(km\^3/s\^2\)\s*=\s*([\d.]+)\s*±?\s*([\d.]+)",
                  r"GM   \(km\^3/s\^2\)\s*=\s*([\d.]+)\s*±?\s*([\d.]+)",
                  r"GM \(km\^3/s\^2\)\s*=\s*([\d.]+)\s*[\+\-]\s*([\d.]+)",
                  r"GM   \(km\^3/s\^2\)\s*=\s*([\d.]+)\s*[\+\-]\s*([\d.]+)",
                  r"GM   \(km\^3/s\^2\)\s*=\s*([\d.]+)\s*±?\s*([\d.]+)"
                  r"GM \(km\^3/s\^2\)\s*=\s*([\d.]+)[+-]\s*([\d.]+)",
                  r"GM\s*\(km\^3/s\^2\)\s*=\s*([\d.]+)\s*[+-]\s*([\d.]+)", 
                  r"GM\s*\(km\^3/s\^2\)\s*=\s*([\d.]+)\s*([+-])\s*([\d.]+)",
                  r"GM\s*\(km\^3/s\^2\)\s*=\s*([\d.]+)",
                  r"GM\s*\(km\^3/s\^2\)\s*=\s*([\d.]+)+-?[\d.]*",
                  r"GM\s*\(km\^3/s\^2\)\s*=\s*([\d.]+)\s*[+-]\s*[\d.]+",
                  r"GM\s*\(?km\^3/s\^2\)?\s*=\s*([\d.]+)",# Matches "GM" with flexible whitespace, allows for [+-] symbols
    ]

    
    mass = None
    for pattern in mass_patterns:
        match = re.search(pattern, raw_data)
        if match:
            if 'kg' in pattern:
                exponent = int(match.group(1))
                base_value = float(match.group(2))
                mass = base_value * (10 ** exponent)  # Mass in kg
            elif 'g' in pattern:
                exponent = int(match.group(1))
                base_value = float(match.group(2)) * 1e-3  # Convert grams to kg
                mass = base_value * (10 ** exponent) 
            break

        if not match:
            #Assign mass for moons based GM value, if no M value is available
            for pattern in gm_pattern:
                match = re.search(pattern, raw_data)
                if match:
                    gm_value = float(match.group(1))
                    mass=gm_value/6.67430e-20
    return mass

def generate_radius(raw_data):
    """
    Function to generate the radius from the raw data
    Needs several radius patterns, as JPL does not use one clear pattern
    """
    radius_patterns = [
        r"Radius \(km\)\s*=\s*([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)",  # Patter
        r"Vol\. Mean Radius \(km\)\s*=\s*([\d.]+)\s*\+\-",
        r"Vol\. mean radius \(km\)\s*=\s*([\d.]+)\s*\+\-",
        r"Vol. Mean Radius \(km\)\s*=\s*([\d.]+)",
        r"Vol\. mean radius km\s*=\s*([\d.]+)\s*\+\-",
        r"Vol. mean radius, km\s*=\s*([\d.]+)\s*\+\-",
        r"Mean radius \(km\)\s*=\s*([\d.]+)",#\s*\+\-",
        r"Mean Radius \(km\)\s*=\s*([\d.]+)",#\s*\+\-",
        r"Radius \(km\)\s*=\s*([\d.]+)",#\s*\+\-",
    ]
    
    radius = None
    for pattern in radius_patterns:
        match = re.search(pattern, raw_data)
        if match:
            if 'x' in pattern:
                # Handle the case where there are three radius values
                radius1 = float(match.group(1))
                radius2 = float(match.group(2))
                radius3 = float(match.group(3))
                radius = (radius1 + radius2 + radius3) / 3 
            else:
                radius = float(match.group(1))  # Radius in km
            break

    return radius

### Functions that combines raw data into dataframe of important parameters ###

def parse_keplerian_data(raw_data, name, date, ID, parent_id=None):
    """
    Function that uses raw JPL input data and transforming it into dataframe
    with only the import parameters: orbital elements, date, mass etc.
    Returns dataframe with parameters of object x"""
    text = raw_data.text
    lines = text.split('\n')
    data_line = None
    in_block = False

    for line in lines:
        if line.startswith("$$SOE"):
            in_block = True
        elif line.startswith("$$EOE"):
            break
        elif in_block and line.strip() and not line.startswith("JD"):
            data_line = line.split(',')
            break

    columns = ['JDTDB', 'Calendar Date (TDB)', 'ecc', 'QR', 'incl', 'longitude',
               'argument', 'Tp', 'N', 'mean_anomaly', 'TA', 'semi', 'AD', 'PR', '0']

    df = pd.DataFrame([data_line], columns=columns)
    df = df.apply(pd.to_numeric, errors='coerce')

    #Adding other parameters to dataframe
    df['mass'] = generate_mass(text)
    df['radius'] = generate_radius(text)
    df['name'] = name
    df['date'] = date
    df['ID'] = int(ID)
    if parent_id is not None:
        df['parent_id'] = parent_id

    #Ignoring non-relevant params
    df.drop(columns=['JDTDB', 'Calendar Date (TDB)', 'QR', 'Tp', 'N', 'TA', 'AD', 'PR', '0'], inplace=True)
    return df.iloc[0].to_dict()

def get_orbital_elements_from_JPL(object_id, name, date, center):
    raw = get_keplerian_elements(object_id, center, date, 'elements')
    return parse_keplerian_data(raw, name, date, object_id)

### Additional code used to check the solarsystem ###

def parse_vector_data(raw_data, name, date, parent_id=None):
    """
    Function that generates vector data (cartesian coordinates) instead of orbital elements
    from JPL, in order to check whether the code works accordingly
    """
    text = raw_data.text
    lines = text.split("\n")
    data_line = None
    in_block = False

    for line in lines:
        if line.startswith("$$SOE"):
            in_block = True
        elif line.startswith("$$EOE"):
            break
        elif in_block and line.strip() and not line.startswith("JD"):
            data_line = line.split(',')
            break

    columns = ["JDTDB", "Calendar Date (TDB)", "X", "Y", "Z", "VX", "VY", "VZ", "LT", "RG", "RR", "0"]
    df = pd.DataFrame([data_line], columns=columns)
    df = df.apply(pd.to_numeric, errors='coerce')

    df['name'] = name
    df['date'] = date
    if parent_id is not None:
        df['parent_id'] = parent_id

    return df.iloc[0].to_dict()