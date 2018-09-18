import pandas as pd
import numpy as np
from ang.header import Header

import skimage
from skimage.transform import PolynomialTransform

ANG_COLUMNS = ['euler1', 'euler2', 'euler3', 'x', 'y', 'iq', 'ci', 'fit', 'phase', 'col9', 'col10', 'col11', 'col12', 'col13']

class Ang(object):

    def __init__(self, ang_path):

        with open(ang_path, "r") as ang_file:

            # Parse header
            header_str = ""
            for line in ang_file:
                if line.startswith('#'):
                    header_str += line
                else:
                    break

            self._header = Header()
            self._header.read(header_str=header_str)

        # We need to re-open the file to allow panda to correctly work
        with open(ang_path, "r") as ang_file:
            self._body = pd.read_csv(ang_file, delim_whitespace=True, comment='#', header=None,
                                     names=ANG_COLUMNS)

    @property
    def header(self):
        return self._header

    @property
    def phases(self):
        return self._header._phases

    @property
    def body(self):
        return self._body

    def warp(self, params, ignore_phase=False):

        for i, name in enumerate(self._body.columns):

            if not ignore_phase and name == 'phase':
                continue

            if name in ['x', 'y']:
                continue

            values = self._body[name].values
            values = np.reshape(values, newshape=self._header.shape)

            transform = skimage.transform._geometric.PolynomialTransform(params)
            values_distord = skimage.transform.warp(values, transform,
                                                    cval=0,
                                                    order=0,  # k-neighbour
                                                    preserve_range=True)

            self._body.iloc[:, i] = np.reshape(values_distord, newshape=-1)

    def add_phase(self, segment_align, phase):

        self._header._phases.append(phase)
        assert len(self._header._phases) == 2

        segment_column = np.concatenate(segment_align, axis=0)
        segment_column = np.copy(segment_column)

        segment_column[np.where(segment_column <= 255 / 2)] = self._header._phases[0].id
        segment_column[np.where(segment_column > 255 / 2)] = self._header._phases[1].id

        self._body["phase"] = segment_column
        self._body["fit"] = segment_column  # Because ANG sucks and decide to put phase in fit...

    def dump_file(self, ang_path):

        with open(ang_path, "w") as ang_file:
            ang_file.write(self._header.dump())

        self._body.to_csv(ang_path,
                          columns=ANG_COLUMNS,
                          index=False,
                          header=False,
                          sep='\t',
                          mode='a',
                          float_format='%.5f')

# class AngWriter:
#     def write_header(self, ang_in, ang_out):
#         with open(ang_in, 'r') as input_ang, open(ang_out, 'w') as output_ang:
#             for line in input_ang:
#                 if line.startswith("#"):
#                     output_ang.write(line)
#
#     def replace_phase(self, segment, ang_in, ang_out):
#         seg = plt.imread(segment)
#         seg_c = np.concatenate(seg, axis=0)  # convert into a column array
#         phase = np.copy(seg_c).astype(int)
#         phase[np.where(seg_c == 255)] = 1
#         phase[np.where(seg_c < 255)] = 2
#         ang = pd.read_csv(ang_in, delim_whitespace=True, comment='#', names={'euler1', 'euler2', 'euler3', 'x',
#                                                                          'y', 'iq', 'ci', 'fit', 'phase',
#                                                                          'col9', 'col10', 'col11', 'col12',
#                                                                          'col13'})
#         ang.iloc[ang_in[:, 6] < 0] = 0  # remove any negative confidence index
#
#         ang.to_csv(ang_out, index=False, header=False, sep='\t', mode='a', float_format='%.5f')
