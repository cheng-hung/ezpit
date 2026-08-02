import os
import re
from collections import Counter
import numpy as np


def parse_composition(composition):
    """
    [EN] Parse a composition string from the Control Panel and turn it into a
         dictionary (EZPDF_GUI_3 helpers.py behaviour).
         Both spaced and compact styles are accepted, and a quantity of 1 may be
         omitted. All of the following give {'Co': 38, 'O': 119, 'P': 1}:
             'Co 38 O 119 P 1'  /  'Co 38 O 119 P'  /  'Co38O119P1'  /  'Co38O119P'
         Fractional amounts are kept AS-IS (not scaled to integers), because
         form-factor averages are normalised by the total, so
             'Li0.2Co0.36Mn0.37Ni0.07'  ==  'Li20Co36Mn37Ni7'
         give identical <f> and <f^2>. A dict input is returned as-is.
    [KR] 조성 문자열을 딕셔너리로 파싱합니다 (EZPDF_GUI_3 helpers.py 방식).
         공백/붙여쓰기 모두 지원하고 개수 1은 생략 가능. 소수 조성은 정수로
         변환하지 않고 그대로 유지합니다 (form-factor 평균이 총합으로 정규화되어
         'Li0.2Co0.36...' 과 'Li20Co36...' 이 동일한 결과를 주기 때문).

    Rules:
      - An element symbol is one uppercase letter optionally followed by one
        lowercase letter (H, O, Si, Co, Mn, ...). Correct capitalisation required.
      - A quantity is a positive number (int or decimal); omitted -> 1.
        Whole numbers are returned as int (existing integer code unchanged).
      - Whitespace anywhere is ignored.
      - A repeated element is summed ('CoOCo' -> {'Co': 2, 'O': 1}).

    Args:
        composition (str or dict): [EN] Composition string or dict / [KR] 조성 문자열 또는 딕셔너리

    Returns:
        collections.Counter / dict: {element: count}  (count may be float if fractional)
    """
    # [EN] Accept a dict directly (already parsed) / [KR] 딕셔너리 입력은 그대로 반환
    if isinstance(composition, dict):
        return composition

    if composition is None or str(composition).strip() == "":
        raise ValueError("The composition field must not be empty")

    # [EN] Drop all whitespace so 'Co 38 O 119' and 'Co38O119' parse identically.
    # [KR] 공백 제거 → 'Co 38 O 119' 와 'Co38O119' 를 동일하게 파싱
    compact = "".join(ch for ch in composition if not ch.isspace())

    composition_dict = Counter()
    i = 0
    length = len(compact)
    while i < length:
        ch = compact[i]

        # [EN] An element symbol must start with an uppercase letter.
        # [KR] 원소 기호는 대문자로 시작해야 함
        if not ch.isupper():
            raise ValueError(
                'Cannot read "{0}" — expected an element symbol starting '
                'with a capital letter (e.g. Co 38 O 119 P 1 or Co38O119P)'.format(ch)
            )

        # [EN] Read symbol: uppercase + optional one lowercase.
        # [KR] 기호 읽기: 대문자 + 선택적 소문자 1개
        element = ch
        i += 1
        if i < length and compact[i].islower():
            element += compact[i]
            i += 1

        # [EN] Read quantity: digits with at most one decimal point, or nothing (=1).
        # [KR] 개수 읽기: 소수점 최대 1개 포함 숫자, 없으면 1
        digits = ""
        seen_dot = False
        while i < length and (compact[i].isdigit() or compact[i] == "."):
            if compact[i] == ".":
                if seen_dot:
                    raise ValueError(
                        "'{0}': count has more than one decimal point".format(element)
                    )
                seen_dot = True
            digits += compact[i]
            i += 1

        if digits == "":
            quantity = 1
        else:
            try:
                quantity = float(digits)
            except ValueError:
                raise ValueError("'{0}': '{1}' is not a valid number".format(element, digits))
            if quantity <= 0:
                raise ValueError(
                    "'{0}': count must be a positive number (got {1})".format(element, digits)
                )
            # [EN] Keep whole numbers as int (existing integer code unchanged).
            # [KR] 정수는 int로 유지 (기존 정수 기반 코드 그대로 동작)
            if float(quantity).is_integer():
                quantity = int(quantity)

        composition_dict[element] += quantity

    if not composition_dict:
        raise ValueError(
            'Invalid composition string "{0}": no element found'.format(composition)
        )

    return composition_dict


def composition_weights(composition):
    """
    [EN] Turn a composition dict into unique element names and their amounts.
         Fraction-safe counterpart of convert_atom_names() + group_atoms().
         Rather than listing one entry per atom, it keeps one entry per unique
         element together with its amount (weight), which is all the form-factor
         averages actually need:
             <f>   = sum_k(c_k * f_k)   / sum_k(c_k)
             <f^2> = sum_k(c_k * f_k^2) / sum_k(c_k)
         Because both averages are normalised by the total, scaling every element
         by the same factor leaves the result unchanged. These all give identical
         averages:
             {'Li': 0.2, 'Co': 0.36, 'Mn': 0.37, 'Ni': 0.07}
             {'Li': 20,  'Co': 36,   'Mn': 37,   'Ni': 7}
             {'Li': 200, 'Co': 360,  'Mn': 370,  'Ni': 70}
    [KR] 조성 딕셔너리를 고유 원소 이름과 그 양(weight)으로 변환합니다.
         convert_atom_names() + group_atoms()의 소수 지원 버전입니다.
         원자 하나당 항목을 만드는 대신, 고유 원소당 항목 하나와 그 양을 유지합니다.
         form-factor 평균은 총합으로 정규화되므로 모든 원소를 같은 배수로 스케일해도
         결과가 동일합니다.

    Args:
        composition (dict or str): [EN] Composition (parsed via parse_composition if str)
                                   [KR] 조성 (문자열이면 parse_composition으로 파싱)

    Returns:
        (names, weights):
            names (list[str])      : [EN] Unique element names / [KR] 고유 원소 이름
            weights (numpy.ndarray): [EN] Their amounts as floats / [KR] 각 원소의 양 (실수)
    """
    # [EN] Parse string input into a dict first / [KR] 문자열이면 먼저 딕셔너리로 파싱
    if not isinstance(composition, dict):
        composition = parse_composition(composition)

    if not composition:
        raise ValueError("The composition must not be empty")

    names = list(composition.keys())
    weights = np.asarray([float(composition[el]) for el in names], dtype=float)

    if np.any(weights <= 0):
        raise ValueError("All composition amounts must be positive")

    return names, weights


def load_atom_name_positions(file_path):
    """
    Parameters
    ==========
    file_path: str
        Path to text file

    Returns
    =======

    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    atom_names = []
    atom_positions = []
    for line in lines:
        parts = line.split()
        atom_names.append(parts[0])
        atom_positions.append([float(x) for x in parts[1:]])
    atom_positions = np.array(atom_positions)
    return atom_names, atom_positions


def load_atom_names(file_path):
    """
    database_atom_names = losa.load_atom_names(compton_aff_element_file)
    print('database_atom_names = ', database_atom_names)
    print all atom names in compton_element_only.txt
    database_atom_names =  ['H', 'He', 'Li', 'Be',,,,,,,,,,,'U']
    """
    with open(file_path, 'r') as f:
        atom_list = [line.strip() for line in f.readlines() if line.strip() != '']
    return atom_list


def load_scattering_factors(file_path):
    """
    database_scat_factors = losa.load_scattering_factors(compton_aff_parm_file)
    print('database_scat_factors = ', database_scat_factors)
    print all parameters for compton scattering form factor in compton_parameter_only.txt
    """
    with open(file_path, 'r') as f:
        data = []
        for line in f:
            data.append([float(val) for val in line.strip().split('\t')])
    return np.array(data)


def convert_atom_names(composition):
    """
    [EN] Convert a whole-number composition (dict OR string) into a full list of
         atom names, one entry per atom. Used by xyz-model / theoretical paths.
         For a composition that may contain FRACTIONAL amounts, use
         composition_weights() instead (fractions cannot be expanded into atoms).
    [KR] 정수 조성(딕셔너리 또는 문자열)을 원자 하나당 한 항목의 리스트로 변환합니다.
         소수 조성은 원자로 확장할 수 없으므로 composition_weights()를 사용하세요.

    Example (dict)  : {'Co': 3, 'O': 4, 'P': 1}  -> ['Co','Co','Co','O','O','O','O','P']
    Example (string): "Co3O4P1" / "Co 3 O 4 P" / "Co3O4P"  -> same as above
    """
    # [EN] Parse string (or accept dict) first / [KR] 문자열이면 먼저 파싱
    comp = parse_composition(composition)

    # [EN] Fractions cannot be expanded into individual atoms.
    # [KR] 소수는 개별 원자로 확장할 수 없음
    for el, count in comp.items():
        if not float(count).is_integer():
            raise ValueError(
                "'{0}': fractional amount {1} cannot be expanded into individual "
                "atoms. Use composition_weights() instead, or scale every element "
                "by the same factor (e.g. Li0.2Co0.36 -> Li20Co36).".format(el, count)
            )

    return [el for el, count in comp.items() for _ in range(int(count))]


def get_scattering_factors(atom_names, database_atom_names,
                           database_scat_factors):
    """
    find Compton scattering parameters according to the given composition
    print('compton_scattering_factors ====== ', Compton_scattering_factors)
    # --> if composition is composition = {'Co':2, 'O':1, 'P':1}, get Compton scattering parameters for Co, O, P
    num_atom = len(atom_indices)
    # num_atom =  4, i.e, # N: total number of atoms in composition
    num_fact = len(compton_scattering_factors)
    # num_fact =  3, i.e. # how many different atoms in composition
    print('atomic_number = ', atomic_number)  # show atomic number of each atom in periodic table
    # --> if composition is composition = {'Co':2, 'O':1, 'P':1},  atomic_number =  [27, 8, 15]
    """
    scat_factors = []
    for atom in atom_names:
        if atom in database_atom_names:
            idx = database_atom_names.index(atom)
            scat_factors.append(database_scat_factors[idx])
        else:
            raise ValueError("There is no atom {0} in database".format(atom))
    return np.asarray(scat_factors)


#### added 12/18/2023######################################
def get_compton_scattering_factors(atom_names, database_atom_names,
                           database_scat_factors):
    """
    add compton scattering factor with give experimental q
    """

    scat_factors = []
    atomic_number = []
    for atom in atom_names:
        if atom in database_atom_names:
            idx = database_atom_names.index(atom)
            scat_factors.append(database_scat_factors[idx])
            atomic_number.append(idx+1)
        else:
            raise ValueError("There is no atom {0} in database".format(atom))
    return np.asarray(scat_factors), atomic_number
#### added 12/18/2023######################################


def group_atoms(atom_names):
    """
    Get unique atom names, their counts, and the index of atom in the unique
    name list. Results will be used by other functions.
            # if composition = {'Co':2, 'O':1, 'P':1},
            counter =  Counter({'Co': 2, 'O': 1, 'P': 1})
            atom_unique_names =  ['Co', 'O', 'P']
            atom_counts =  [2 1 1]  # "2" is for 'Co', "1" is for 'O', "1" is for 'P'
            atom_indices  =  [0 0 1 2] #
    """
    counter = Counter(atom_names)
    atom_uni_names = list(counter.keys())
    atom_counts = list(counter.values())
    atom_indices = [atom_uni_names.index(atom) for atom in atom_names]
    return atom_uni_names, np.asarray(atom_counts), np.asarray(atom_indices)


def make_folder(file_path):
    """
    Create a folder for saving file if the folder does not exist. This is a
    supplementary function for savers.
    Parameters
    ----------
    file_path : str
        Path to a file.
    """
    file_base = os.path.dirname(file_path)
    if not os.path.exists(file_base):
        try:
            os.makedirs(file_base, exist_ok=True)
        except OSError:
            raise ValueError("Can't create the folder: {}".format(file_base))


def save_txt(filename, q_Iq):
    make_folder(filename)
    np.savetxt(filename, q_Iq)