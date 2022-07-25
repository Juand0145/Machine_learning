#!/usr/bin/env python3
"""File that contain sthe fuctionssentientPlanets """
import requests


def sentientPlanets():
    """
    Function that  a method that returns the list of names of the
    home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    speciesList = []
    while url:
        results = requests.get(url).json()
        speciesList += results.get('results')
        url = results.get('next')
    homePlanets = []
    for species in speciesList:
        if species.get('designation') == 'sentient' or \
           species.get('classification') == 'sentient':
            url = species.get('homeworld')
            if url:
                planet = requests.get(url).json()
                homePlanets.append(planet.get('name'))
    return homePlanets
