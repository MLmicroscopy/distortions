import re
import io

from ang.phase import Phase


class Header(object):

    def __init__(self):
        self._pix_per_um = None
        self._x_star = None
        self._y_star = None
        self._z_star = None
        self._working_distance = None
        self._grid = None
        self._x_step = None
        self._y_step = None
        self._n_col_odd = None
        self._n_col_even = None
        self._n_rows = None
        self._phases = []

    @property
    def shape(self):
        return [self._n_rows, self._n_col_even]

    @property
    def phases(self):
        return self._phases

    @shape.setter
    def shape(self, val):
        self._n_rows = val[0]
        self._n_col_even = val[1]
        self._n_col_odd = val[1]

    def read(self, header_str):

        lines = header_str.splitlines()

        reading_phase, phase_str = False, ""

        for i, line in enumerate(lines):

            if line.startswith("#"):
                tokens = re.split('\s+', line.strip())

                if len(tokens) == 1:
                    if reading_phase:
                        phase = Phase.load_from_string(phase_str)
                        self._phases.append(phase)
                        reading_phase, phase_str = False, ""
                    continue
                else:
                    field = tokens[1]

                if field == 'TEM_PIXperUM':
                    self._pix_per_um = float(tokens[2])

                elif field == 'x-star':
                    self._x_star = float(tokens[2])

                elif field == 'y-star':
                    self._y_star = float(tokens[2])

                elif field == 'z-star':
                    self._z_star = float(tokens[2])

                elif field == 'WorkingDistance':
                    self._working_distance = float(tokens[2])

                elif field == 'GRID:':
                    self._grid = tokens[2]

                elif field == 'XSTEP:':
                    self._x_step = float(tokens[2])

                elif field == 'YSTEP:':
                    self._y_step = float(tokens[2])

                elif field == 'NCOLS_ODD:':
                    self._n_col_odd= int(tokens[2])

                elif field == 'NCOLS_EVEN:':
                    self._n_col_even = int(tokens[2])

                elif field == 'NROWS:':
                    self._n_rows = int(tokens[2])

                elif field == 'Phase' or reading_phase:
                    reading_phase = True
                    phase_str += line
                    phase_str += "\n"


            else:
                print("Finish parsing header!")
                return True

    def dump(self):

        with io.StringIO() as buffer:

            # Dump upper header
            buffer.write("# TEM_PIXperUM\t{:.6f}\n".format(self._pix_per_um))
            buffer.write("# x-star\t{:.6f}\n".format(self._x_star))
            buffer.write("# y-star\t{:.6f}\n".format(self._y_star))
            buffer.write("# z-star\t{:.6f}\n".format(self._z_star))
            buffer.write("# WorkingDistance\t{:.6f}\n".format(self._working_distance))

            # Dump phase
            buffer.write("#\n")
            for phase in self._phases:
                buffer.write(phase.dump())
                buffer.write("#\n")

            # Dump lower header
            buffer.write("# GRID: {}\n".format(self._grid))
            buffer.write("# XSTEP: {:.6f}\n".format(self._x_step))
            buffer.write("# YSTEP: {:.6f}\n".format(self._y_step))
            buffer.write("# NCOLS_ODD: {}\n".format(self._n_col_odd))
            buffer.write("# NCOLS_EVEN: {}\n".format(self._n_col_even))
            buffer.write("# NROWS: {}\n".format(self._n_rows))
            buffer.write("#\n")
            buffer.write("# OPERATOR: \tcustomerservice\n")
            buffer.write("#\n")
            buffer.write("# SAMPLEID: \t\n")
            buffer.write("#\n")
            buffer.write("# SCANID: \t\n")
            buffer.write("#\n")

            sheader = buffer.getvalue()

        return sheader











