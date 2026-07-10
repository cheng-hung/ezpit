# This file was made so that we don't need to reload the data everytime an API call to calculator.py is made.
import pandas as pd
import numpy as np


class ElementData:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):  # Prevent re-initialization
            return

        self.aff_element = pd.read_csv('elem_info/aff_elementonly.txt', header=None, names=['element'])
        self.aff_element_dict = dict(zip(self.aff_element['element'].str.lower(), self.aff_element.index))
        self.aff_parm = pd.read_csv('elem_info/aff_parmonly.txt', sep='\t', header=None)

        self.compton_atomic_number = pd.read_csv('elem_info/compton_atomicnumber.txt', header=None,
                                                 names=['atomic_number'])
        self.compton_element = pd.read_csv('elem_info/compton_element_only.txt', header=None, names=['element'])
        self.compton_element_dict = dict(zip(self.compton_element['element'].str.lower(), self.compton_element.index))

        self.compton_parameter_only = pd.read_csv('elem_info/compton_parameter_only.txt', sep='\t', header=None)

        ElementData._instance = self

    def aff_element_to_index(self):
        return self.aff_element_dict

    def compton_element_to_index(self):
        return self.compton_element_dict

    def get_aff_scattering_factors(self, atom_names):
        """
        Returns atomic form factor parameters for given atoms.
        """
        return np.array([self.aff_parm.iloc[self.aff_element_dict[name.lower()]] for name in atom_names])

    def get_compton_scattering_factors(self, atom_names):
        """
        Returns Compton scattering parameters and atomic numbers for given atoms.
        """
        scat_factors = []
        atomic_numbers = []
        for name in atom_names:
            idx = self.compton_element_dict.get(name.lower())
            if idx is None:
                raise ValueError(f"No Compton data for atom: {name}")
            scat_factors.append(self.compton_parameter_only.iloc[idx])
            atomic_numbers.append(self.compton_atomic_number.iloc[idx, 0])
        return np.array(scat_factors), atomic_numbers

    def get_compton_parameter_only(self):
        return np.array(self.compton_parameter_only)
