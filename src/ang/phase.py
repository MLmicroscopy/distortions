import numpy as np
import io
import re

import collections


def np2str(arr, precision=6, smart_int=False):
    assert len(arr.shape) == 1

    s = []
    for x in arr:
        x = float(x)
        if smart_int and x.is_integer():
            s += ["{}".format(int(x))]
        else:
            s += ["{0:.{1}f}".format(x, precision)]

    return " ".join(s)


_default_latticeConstants = np.zeros([6])  # 3 constants & 3 angles
_default_elasticConstants = np.zeros([6, 6])
_default_categories = np.zeros([4])

class Phase(object):
    def __init__(self, id,
                 name='',
                 formula='',
                 symmetry=0,
                 latticeConstants=_default_latticeConstants,
                 hklFamilies=np.array([]),
                 elasticConstants=np.array([]),
                 categories=_default_categories):

        self._id = id
        self._materialName = name
        self._formula = formula
        self._info = ''
        self._symmetry = symmetry
        self._latticeConstants = latticeConstants
        self._hklFamilies = hklFamilies
        self._elasticConstants = elasticConstants
        self._categories = categories

    @property
    def id(self):
        return self._id

    @property
    def materialName(self):
        return self._materialName

    @materialName.setter
    def materialName(self, val):
        if not isinstance(val, str):
            raise AttributeError("materialName must be a string.")
        self._materialName = val

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, val):
        if not isinstance(val, str):
            raise AttributeError("formula must be a string.")
        self._formula = val

    @property
    def symmetry(self):
        return self._symmetry

    @symmetry.setter
    def symmetry(self, val):
        if not isinstance(val, int) or val < 0:
            raise AttributeError("symmetry must be a positive integer. Get {}".format(val))
        self._symmetry = val

    @property
    def latticeConstants(self):
        return self._latticeConstants

    @latticeConstants.setter
    def latticeConstants(self, val):
        if not isinstance(val, np.ndarray):
            raise AttributeError("latticeConstants must be a numpy array")
        if val.shape != _default_latticeConstants.shape:
            raise AttributeError("latticeConstants must have a shape of {}. Get {}"
                                 .format(_default_latticeConstants.shape, val.shape))
        self._latticeConstants = val

    @property
    def hklFamilies(self):
        return self._hklFamilies

    @hklFamilies.setter
    def hklFamilies(self, val):
        if not isinstance(val, np.ndarray):
            raise AttributeError("hklFamilies must be a numpy array")
        if len(val.shape) not in [0, 2] or val.shape[1] != 6:
            raise AttributeError("hklFamilies must have a shape of ?x6. Get {}".format(val.shape))

        self._hklFamilies = val

    @property
    def numberFamilies(self):
        return len(self._hklFamilies)

    @property
    def elasticConstants(self):
        return self._elasticConstants

    @elasticConstants.setter
    def elasticConstants(self, val):
        if not isinstance(val, np.ndarray):
            raise AttributeError("elasticConstants must be a numpy array")
        if len(val.shape) not in [0, 2] or val.shape != _default_elasticConstants.shape:
            raise AttributeError("elasticConstants must have a shape of {}. Get {}"
                                 .format(_default_elasticConstants.shape, val.shape))
        self._elasticConstants = val

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, val):
        if not isinstance(val, np.ndarray):
            raise AttributeError("categories must be a numpy array")
        if val.shape != _default_categories.shape:
            raise AttributeError("categories must have a shape of {}. Get {}"
                                 .format(_default_categories.shape, val.shape))
        self._categories = val

    @staticmethod
    def create_from_phase(id, name, formula, phase):
        return Phase(id=id,
                     name=name,
                     formula=formula,
                     symmetry=phase.symmetry,
                     latticeConstants=phase.latticeConstants,
                     hklFamilies=phase.hklFamilies,
                     elasticConstants=phase.elasticConstants,
                     categories=phase.categories)

    @staticmethod
    def load_from_string(sphase):

        lines = sphase.splitlines()

        # extract phase id (compulsary)
        phase_id = re.findall("# Phase (\d+)", string=lines[0])
        if len(phase_id) != 1:
            raise Exception("Invalid Format: Could not retrieve the phase id")

        phase = Phase(id=int(phase_id[0]))

        list_buffer = collections.defaultdict(list)

        for i, line in enumerate(lines[1:]):

            if line.startswith("#"):
                tokens = re.split('\s+', line.strip())
                if tokens[1] == 'MaterialName':
                    phase.materialName = str(tokens[2])
                elif tokens[1] == 'Formula':
                    phase.formula = str(tokens[2])
                elif tokens[1] == 'Symmetry':
                    phase.symmetry = int(tokens[2])
                elif tokens[1] == 'LatticeConstants':
                    phase.latticeConstants = np.array(tokens[2:], dtype=np.float32)
                elif tokens[1] == 'Categories0':
                    phase.categories = np.array(tokens[2:], dtype=np.float32)
                elif tokens[1] in ['hklFamilies',
                                   'ElasticConstants']:
                    list_buffer[tokens[1]].append(tokens[2:])

            else:
                raise Exception("Invalid Format: missing diese in the header")

        # Post process list
        if "hklFamilies" in list_buffer:
            phase.hklFamilies = np.array(list_buffer["hklFamilies"], dtype=np.float32)

        if "ElasticConstants" in list_buffer:
            phase.elasticConstants = np.array(list_buffer["ElasticConstants"], dtype=np.float32)

        return phase

    def dump(self):

        with io.StringIO() as buffer:
            buffer.write("# Phase\t{}\n".format(self.id))
            buffer.write("# MaterialName\t{}\n".format(self.materialName))
            buffer.write("# Formula\t{}\n".format(self.formula))
            buffer.write("# Info\n")
            buffer.write("# Symmetry\t{}\n".format(self.symmetry))
            buffer.write("# LatticeConstants\t{}\n".format(np2str(self.latticeConstants, precision=3)))
            buffer.write("# NumberFamilies\t{}\n".format(self.numberFamilies))
            for family in self.hklFamilies:
                buffer.write("# hklFamilies \t{}\n".format(np2str(family, smart_int=True)))
            for elastic in self.elasticConstants:
                buffer.write("# ElasticConstants \t{}\n".format(np2str(elastic, precision=6)))
            buffer.write("# Categories0\t{}\n".format(np2str(self.categories, smart_int=True)))

            sphase = buffer.getvalue()

        return sphase


