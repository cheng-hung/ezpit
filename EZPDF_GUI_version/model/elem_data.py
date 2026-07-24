# This file loads the element data files once so that they do not need to be
# re-read on every calculation.
import os
import sys

import pandas as pd
import numpy as np


def resource_path(relative_path):
    """Return the absolute path to a bundled data file.

    Data files must be located relative to the program itself, not relative to
    the directory the user happens to launch it from. Using a bare relative
    path such as 'elem_info/aff_elementonly.txt' resolves against the current
    working directory, so the files are not found whenever the program is
    started from anywhere else -- which is always the case for a packaged
    executable that the user double-clicks.

    This helper handles both situations:

    * Running from source: the path is resolved relative to the project root,
      i.e. the parent directory of the folder holding this file.
    * Running from a PyInstaller bundle: PyInstaller sets ``sys._MEIPASS`` to
      the folder containing the bundled data files, so that is used instead.
    """
    if hasattr(sys, "_MEIPASS"):
        # Running inside a PyInstaller bundle.
        base_path = sys._MEIPASS
    else:
        # Running from source: model/ -> project root.
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ElementData:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):  # Prevent re-initialization
            return

        self.aff_element = pd.read_csv(
            resource_path('elem_info/aff_elementonly.txt'), header=None, names=['element'])
        self.aff_element_dict = dict(zip(self.aff_element['element'].str.lower(), self.aff_element.index))
        self.aff_parm = pd.read_csv(
            resource_path('elem_info/aff_parmonly.txt'), sep='\t', header=None)

        self.compton_atomic_number = pd.read_csv(
            resource_path('elem_info/compton_atomicnumber.txt'), header=None,
            names=['atomic_number'])
        self.compton_element = pd.read_csv(
            resource_path('elem_info/compton_element_only.txt'), header=None, names=['element'])
        self.compton_element_dict = dict(zip(self.compton_element['element'].str.lower(), self.compton_element.index))

        self.compton_parameter_only = pd.read_csv(
            resource_path('elem_info/compton_parameter_only.txt'), sep='\t', header=None)

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