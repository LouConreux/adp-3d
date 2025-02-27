"""
\file Profile.cpp   \brief A class for profile storing and computation

Copyright 2007-2022 IMP Inventors. All rights reserved.
"""

import copy
import math
import numpy as np
from numba import jit
import torch

from src.structure import get_default_form_factor_table, FormFactorType
from src.function import SincFunction, ExpFunction
from src.distribution import RadialDistributionFunction, get_index_from_distance, get_distance_from_index
from src.solvent import SolventAccessibleSurface
from src.score import ChiScore

IMP_SAXS_DELTA_LIMIT = 1.0e-15

class Profile:
    modulation_function_parameter_ = 0.23

    def __init__(self, qmin=0, qmax=0, delta=1, file_name="", fit_file=True,
        max_q=0, units=1, constructor=0):
        if constructor == 0:
            self.min_q_ = qmin
            self.max_q_ = qmax
            self.delta_q_ = delta
            self.c1_ = 10
            self.c2_ = 10
            self.experimental_ = False
            self.average_radius_ = 1.58
            self.average_volume_ = 17.5
            self.ff_table_ = get_default_form_factor_table()
        else:
            self.experimental_ = True
            if fit_file:
                self.experimental_ = False
            self.read_SAXS_file(file_name, fit_file, max_q, units)
        self.id_ = 0
        self.name_ = file_name
        self.q_mapping_ = {}
        self.beam_profile_ = None
        self.partial_profiles_ = []
        # self.init()

    def init(self, size=0, partial_profiles_size=0):
        number_of_q_entries = size
        if size == 0:
            number_of_q_entries = int(np.ceil((self.max_q_ - self.min_q_) / self.delta_q_)) + 1

        if not hasattr(self, "q_"):
            self.q_ = np.zeros(number_of_q_entries, dtype=np.double)
        if not hasattr(self, "intensity_"):
            self.intensity_ = np.zeros(number_of_q_entries, dtype=np.double)
        if not hasattr(self, "error_"):
            self.error_ = np.zeros(number_of_q_entries, dtype=np.double)

        if size == 0:
            for i in range(number_of_q_entries):
                self.q_[i] = self.min_q_ + i * self.delta_q_

        if partial_profiles_size > 0 and len(self.partial_profiles_) == 0:
            self.partial_profiles_ = [np.zeros(number_of_q_entries,
                dtype=np.double) for _ in range(partial_profiles_size)]

    def find_max_q(self, file_name):
        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()

        max_q = 0.0
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line == '' or not line[0].isdigit():
                continue
            split_results = line.split()
            if len(split_results) < 2 or len(split_results) > 5:
                continue
            if not split_results[0].replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).replace('E', '', 1).replace('e', '', 1).isdigit():
                continue
            max_q = float(split_results[0])

        return max_q

    def background_adjust(self, start_q):
        data = []  # x=q^2, y=sum(q^2xI(q))
        sum_val = 0.0
        for i in range(self.size()):
            q = self.q_[i]
            Iq = self.intensity_[i]
            q2xIq = q * q * Iq
            sum_val += q2xIq
            if q >= start_q:
                v = (q * q, sum_val)
                data.append(v)

        if len(data) == 0:
            print("No points in profile at or above start_q; no background adjustment done")
            return

        # Calculate the parabolic fit coefficients
        x = np.array([item[0] for item in data])
        y = np.array([item[1] for item in data])
        coefficients = np.polyfit(x, y, 2)
        P3, P2, P1 = coefficients

        G1 = P2 / P1
        G2 = 12.0 * (P3 / P1 - G1 * G1 / 4.0)

        for i in range(len(self.q_)):
            q = self.q_[i]
            q2 = q * q
            q4 = q2 * q2
            Iq = self.intensity_[i]
            Iq_new = Iq / (1.0 + q2 * G1 + q4 * (G1 * G1 / 4.0 + G2 / 12.0))
            self.intensity_[i] = Iq_new

    def read_SAXS_file(self, file_name, fit_file, max_q, units):
        with open(file_name, 'r') as in_file:
            lines = in_file.readlines()

        default_units = True
        if units == 3:
            default_units = False
        if units == 1 and self.find_max_q(file_name) > 1.0:
            default_units = False

        with_error = False
        qs = []
        intensities = []
        errors = []

        for line in lines:
            line = line.strip()
            if line.startswith('#') or line == '' or not line[0].isdigit():
                continue
            split_results = line.split()
            if len(split_results) < 2 or len(split_results) > 5:
                continue
            if not split_results[0].replace('.', '', 1).replace('-', '', 1).replace('+', '', 1).replace('E', '', 1).replace('e', '', 1).isdigit():
                continue

            q = float(split_results[0])
            if not default_units:
                q /= 10.0

            if max_q > 0.0 and q > max_q:
                break

            if fit_file:
                if len(split_results) != 4:
                    continue
                intensity = float(split_results[3])
            else:
                intensity = float(split_results[1])

            if abs(intensity) < IMP_SAXS_DELTA_LIMIT:
                continue
            if intensity < 0.0:
                print("Negative intensity value: " + line + " skipping remaining profile points" + "\n")
                # break
            error = 1.0
            if len(split_results) >= 3 and not fit_file:
                error = float(split_results[2])
                if abs(error) < IMP_SAXS_DELTA_LIMIT:
                    error = 0.05 * intensity
                    if abs(error) < IMP_SAXS_DELTA_LIMIT:
                        continue
                with_error = True

            qs.append(q)
            intensities.append(intensity)
            errors.append(error)

        if len(qs) > 0:
            self.init(len(qs))
        for i, e in enumerate(qs):
            self.q_[i] = e
            self.intensity_[i] = intensities[i]
            self.error_[i] = errors[i]

        if self.size() > 1:
            self.min_q_ = self.q_[0]
            self.max_q_ = self.q_[-1]
            if self.is_uniform_sampling():
                diff = 0.0
                for i in range(1, self.size()):
                    diff += self.q_[i] - self.q_[i - 1]
                self.delta_q_ = diff / (self.size() - 1)
            else:
                self.delta_q_ = (self.max_q_ - self.min_q_) / (self.size() - 1)

        print("read_SAXS_file: " + file_name + " size= " + str(self.size()) +
                    " delta= " + str(self.delta_q_) + " min_q= " + str(self.min_q_) +
                    " max_q= " + str(self.max_q_) + "\n")

        if not with_error:
            self.add_errors()
            print("read_SAXS_file: No experimental error specified" +
                        " -> error added " + "\n")

    def add_errors(self):
        for i in range(self.size()):
            ra = abs(np.random.poisson(10.0) / 10.0 - 1.0) + 1.0
            self.error_[i] = 0.03 * self.intensity_[i] * 5.0 * (self.q_[i] + 0.001) * ra
        self.experimental_ = True

    def add_noise(self, percentage):
        for i in range(self.size()):
            random_number = np.random.poisson(10.0) / 10.0 - 1.0
            self.intensity_[i] += percentage * self.intensity_[i] * (self.q_[i] + 1.0) * random_number

    def is_uniform_sampling(self):
        if self.size() <= 1:
            return False
        curr_diff = self.q_[1] - self.q_[0]
        epsilon = curr_diff / 10
        for i in range(2, self.size()):
            diff = self.q_[i] - self.q_[i - 1]
            if abs(curr_diff - diff) > epsilon:
                return False
        return True

    def write_SAXS_file(self, file_name, max_q=None):
        with open(file_name, 'w') as out_file:
            if max_q is None:
                max_q = self.max_q_
            out_file.write("# SAXS profile: number of points = {0}, q_min = {1}, q_max = {2}, delta_q = {3}\n".format(
                self.size(), self.min_q_, max_q, self.delta_q_))
            out_file.write("#    q    intensity ")
            if self.experimental_:
                out_file.write("   error")
            out_file.write("\n")

            out_file.write("{:.8f} {:.8f}".format(self.q_[0], self.intensity_[0]))
            if self.experimental_:
                out_file.write(" {:.8f}".format(self.error_[0]))
            out_file.write("\n")

            for i in range(1, self.size()):
                if self.q_[i] > max_q > 0:
                    break
                out_file.write("{:.8f} {:.8f}".format(self.q_[i], self.intensity_[i]))
                if self.experimental_:
                    out_file.write(" {:.8f}".format(self.error_[i]))
                out_file.write("\n")

    def read_partial_profiles(self, file_name):
        with open(file_name, 'r') as in_file:
            qs = []
            partial_profiles = [[] for _ in range(6)]
            psize = 6

            line = in_file.readline().strip()
            while line:
                if line[0] != '#' and line[0] != '\0' and line[0].isdigit():
                    split_results = line.split()
                    if len(split_results) == 7:
                        qs.append(float(split_results[0]))
                        for i in range(psize):
                            partial_profiles[i].append(float(split_results[i + 1]))

                line = in_file.readline().strip()

        if qs:
            self.init(len(qs), psize)
            for i, e in enumerate(qs):
                self.q_[i] = e
                self.intensity_[i] = 1
                self.error_[i] = 1
                for j in range(psize):
                    self.partial_profiles_[j][i] = partial_profiles[j][i]
        self.sum_partial_profiles(1.0, 0.0, False)

        if self.size() > 1:
            self.min_q_ = self.q_[0]
            self.max_q_ = self.q_[self.size() - 1]

            if self.is_uniform_sampling():
                diff = 0.0
                for i in range(1, self.size()):
                    diff += self.q_[i] - self.q_[i - 1]
                self.delta_q_ = diff / (self.size() - 1)
            else:
                self.delta_q_ = (self.max_q_ - self.min_q_) / (self.size() - 1)

        print("read_partial_profiles: {0} size= {1} delta= {2} min_q= {3} max_q= {4}\n".format(
            file_name, self.size(), self.delta_q_, self.min_q_, self.max_q_))

    def write_partial_profiles(self, file_name):
        with open(file_name, 'w') as out_file:
            # header line
            out_file.write("# SAXS profile: number of points = {}\n".format(self.size()))
            out_file.write("# q_min = {}, q_max = {}, delta_q = {}\n".format(self.min_q_, self.max_q_, self.delta_q_))
            out_file.write("#    q    intensity\n")

            out_file.write('\n'.join(["{:.5f} {:.8f}".format(self.q_[i], self.intensity_[i]) for i in range(self.size())]))

            if len(self.partial_profiles_) > 0:
                for j in range(len(self.partial_profiles_)):
                    out_file.write('\n')
                    out_file.write(' '.join(["{:.8f}".format(self.partial_profiles_[j][i]) for i in range(self.size())]))

        out_file.close()

    def calculate_profile_real(self, particles, ff_type):
        print("start real profile calculation for {} particles\n".format(len(particles)))
        r_dist = RadialDistributionFunction()  # fi(0) fj(0)
        coordinates = np.array([particle.coordinates for particle in particles])
        form_factors = np.array([self.ff_table_.get_form_factor(particle, ff_type) for particle in particles])


        distribution = inner_calculate_profile_real(coordinates, form_factors,
            r_dist.one_over_bin_size)

        r_dist.add_distribution(distribution)

        self.squared_distribution_2_profile(r_dist)

    def old_calculate_profile_real(self, particles, ff_type):
        print("start real profile calculation for {} particles\n".format(len(particles)))
        r_dist = RadialDistributionFunction()  # fi(0) fj(0)
        coordinates = [particle.coordinates for particle in particles]
        form_factors = [self.ff_table_.get_form_factor(particle, ff_type) for particle in particles]

        # iterate over pairs of atoms
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = get_squared_distance(coordinates[i], coordinates[j])
                prod = form_factors[i] * form_factors[j]
                r_dist.add_to_distribution(dist, 2 * prod)

            # add autocorrelation part
            r_dist.add_to_distribution(0.0, math.pow(form_factors[i], 2))

        self.squared_distribution_2_profile(r_dist)

    def calculate_profile_real_gpu(self, particles, ff_type):
        print("start real profile calculation for {} particles\n".format(len(particles)))
        r_dist = RadialDistributionFunction()  # fi(0) fj(0)
        coordinates = torch.tensor([particle.coordinates for particle in particles])
        form_factors = torch.tensor([self.ff_table_.get_form_factor(particle, ff_type)
            for particle in particles])

        distribution = inner_calculate_profile_real_gpu(coordinates, form_factors,
            r_dist.one_over_bin_size)

        r_dist.add_distribution(distribution)

        self.squared_distribution_2_profile(r_dist)

    def calculate_I0(self, particles, ff_type):
        I0 = 0.0
        for particle in particles:
            I0 += self.ff_table_.get_vacuum_form_factor(particle, ff_type)
        return math.pow(I0, 2)

    def calculate_profile_constant_form_factor(self, particles, form_factor):
        print("start real profile calculation for {} particles\n".format(len(particles)))
        r_dist = RadialDistributionFunction()
        coordinates = [particle.coordinates for particle in particles]
        ff = 1 # np.square(form_factor)

        # iterate over pairs of atoms
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = get_squared_distance(coordinates[i], coordinates[j])
                r_dist.add_to_distribution(dist, 2 * ff)

            # add autocorrelation part
            r_dist.add_to_distribution(0.0, ff)

        self.squared_distribution_2_profile(r_dist)

    def calculate_profile_partial(self, particles, surface, ff_type):
        print("start real partial profile calculation for {} particles\n".format(len(particles)))
        coordinates = np.array([particle.coordinates for particle in particles])
        vacuum_ff = np.array([self.ff_table_.get_vacuum_form_factor(particle,
            ff_type) for particle in particles])
        dummy_ff = np.array([self.ff_table_.get_dummy_form_factor(particle,
            ff_type) for particle in particles])

        r_size = 3

        if len(surface) == len(particles):
            r_size = 6
            water_ff = np.array([surface[i]*self.ff_table_.get_water_form_factor()
                for i in range(len(particles))])

        r_dist = [RadialDistributionFunction() for _ in range(r_size)]

        distributions = inner_calculate_profile_partial(coordinates,
            vacuum_ff, dummy_ff, water_ff, r_size, r_dist[0].one_over_bin_size)

        for i, d in enumerate(r_dist):
            d.add_distribution(distributions[i])

        # convert to reciprocal space
        self.squared_distributions_2_partial_profiles(r_dist)

        # compute default profile c1 = 1, c2 = 0
        self.sum_partial_profiles(1.0, 0.0, False)

    def old_calculate_profile_partial(self, particles, surface, ff_type):
        print("start real partial profile calculation for {} particles\n".format(len(particles)))
        coordinates = [particle.coordinates for particle in particles]
        vacuum_ff = [self.ff_table_.get_vacuum_form_factor(particle, ff_type) for particle in particles]
        dummy_ff = [self.ff_table_.get_dummy_form_factor(particle, ff_type) for particle in particles]
        water_ff = None

        if len(surface) == len(particles):
            water_ff = [surface[i] * self.ff_table_.get_water_form_factor() for i in range(len(particles))]

        r_size = 3
        if len(surface) == len(particles):
            r_size = 6

        r_dist = [RadialDistributionFunction() for _ in range(r_size)]

        # iterate over pairs of atoms
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = get_squared_distance(coordinates[i], coordinates[j])
                r_dist[0].add_to_distribution(dist, 2 * vacuum_ff[i] * vacuum_ff[j])  # constant
                r_dist[1].add_to_distribution(dist, 2 * dummy_ff[i] * dummy_ff[j])  # c1^2
                r_dist[2].add_to_distribution(dist, 2 * (vacuum_ff[i] * dummy_ff[j] + vacuum_ff[j] * dummy_ff[i]))  # -c1

                if r_size > 3:
                    r_dist[3].add_to_distribution(dist, 2 * water_ff[i] * water_ff[j])  # c2^2
                    r_dist[4].add_to_distribution(dist, 2 * (vacuum_ff[i] * water_ff[j] + vacuum_ff[j] * water_ff[i]))  # c2
                    r_dist[5].add_to_distribution(dist, 2 * (water_ff[i] * dummy_ff[j] + water_ff[j] * dummy_ff[i]))  # -c1*c2

            # add autocorrelation part
            r_dist[0].add_to_distribution(0.0, vacuum_ff[i] * vacuum_ff[i])
            r_dist[1].add_to_distribution(0.0, dummy_ff[i] * dummy_ff[i])
            r_dist[2].add_to_distribution(0.0, 2 * vacuum_ff[i] * dummy_ff[i])

            if r_size > 3:
                r_dist[3].add_to_distribution(0.0, water_ff[i] * water_ff[i])
                r_dist[4].add_to_distribution(0.0, 2 * vacuum_ff[i] * water_ff[i])
                r_dist[5].add_to_distribution(0.0, 2 * water_ff[i] * dummy_ff[i])

        # convert to reciprocal space
        self.squared_distributions_2_partial_profiles(r_dist)

        # compute default profile c1 = 1, c2 = 0
        self.sum_partial_profiles(1.0, 0.0, False)

    def sum_partial_profiles(self, c1, c2, check_cached=False):
        # precomputed exp function
        # ef = ExpFunction(self.max_q_ * self.max_q_ * 0.3, 0.00001)

        if len(self.partial_profiles_) == 0:
            return

        # check if the profiles are already summed by this c1/c2 combination
        if check_cached and abs(self.c1_ - c1) <= 0.000001 and abs(self.c2_ - c2) <= 0.000001:
            return

        rm = self.average_radius_
        coefficient = -rm * rm * (c1 * c1 - 1.0) / (4 * math.pi)
        square_c2 = c2 * c2
        cube_c1 = c1 * c1 * c1

        self.intensity_ = self.partial_profiles_[0].copy()

        if len(self.partial_profiles_) > 3:
            self.intensity_ += square_c2 * self.partial_profiles_[3]
            self.intensity_ += c2 * self.partial_profiles_[4]

        x = coefficient*np.square(self.q_)
        x_idx = np.fabs(x) > 1.0e-8
        G_q = np.full_like(x, cube_c1)
        G_q[x_idx] *= np.exp(x[x_idx])

        self.intensity_ += G_q * G_q * self.partial_profiles_[1]
        self.intensity_ -= G_q * self.partial_profiles_[2]

        if len(self.partial_profiles_) > 3:
            self.intensity_ -= G_q * c2 * self.partial_profiles_[5]

        # cache new c1/c2 values
        self.c1_ = c1
        self.c2_ = c2

    def old_sum_partial_profiles(self, c1, c2, check_cached=False):
        # precomputed exp function
        ef = ExpFunction(self.max_q_ * self.max_q_ * 0.3, 0.00001)

        if not hasattr(self, "partial_profiles_") or len(self.partial_profiles_) == 0:
            return

        # check if the profiles are already summed by this c1/c2 combination
        if check_cached and abs(self.c1_ - c1) <= 0.000001 and abs(self.c2_ - c2) <= 0.000001:
            return

        rm = self.average_radius_
        coefficient = -rm * rm * (c1 * c1 - 1.0) / (4 * math.pi)
        square_c2 = c2 * c2
        cube_c1 = c1 * c1 * c1

        self.intensity_ = copy.copy(self.partial_profiles_[0])

        if len(self.partial_profiles_) > 3:
            self.intensity_ += square_c2 * self.partial_profiles_[3]
            self.intensity_ += c2 * self.partial_profiles_[4]

        for k in range(self.size()):
            q = self.q_[k]
            x = coefficient * q * q
            G_q = cube_c1
            if abs(x) > 1.0e-8:
                G_q *= ef.exp(x)

            self.intensity_[k] += G_q * G_q * self.partial_profiles_[1][k]
            self.intensity_[k] -= G_q * self.partial_profiles_[2][k]

            if len(self.partial_profiles_) > 3:
                self.intensity_[k] -= G_q * c2 * self.partial_profiles_[5][k]

        # cache new c1/c2 values
        self.c1_ = c1
        self.c2_ = c2


    def resample(self, exp_profile, resampled_profile):
        resampled_profile.init(exp_profile.size(), len(self.partial_profiles_))

        q_rs = resampled_profile.q_
        intensity_rs = resampled_profile.intensity_
        pp_rs = resampled_profile.partial_profiles_

        q_rs, intensity_rs, pp_rs = inner_resample(q_rs, intensity_rs, np.array(pp_rs),
            exp_profile.q_, self.q_, self.intensity_, self.max_q_, self.name_,
            np.array(self.partial_profiles_))

        resampled_profile.q_ = q_rs
        resampled_profile.intensity_ = intensity_rs
        resampled_profile.partial_profiles_ = [pp_rs[i] for i in range(len(pp_rs))]
        # resampled_profile.partial_profiles_ = pp_rs

    def old_resample(self, exp_profile, resampled_profile):
        if not self.q_mapping_:
            for k in range(self.size()):
                self.q_mapping_[self.q_[k]] = k
        # Initialize
        size_pp = len(self.partial_profiles_) if hasattr(self, "partial_profiles_") else 0
        resampled_profile.init(exp_profile.size(), size_pp)

        for k in range(exp_profile.size()):
            q = exp_profile.q_[k]
            q_mapping_iterator = next((it for it in self.q_mapping_.items() if it[0] >= q), None)
            # In case the experimental profile is longer than the computed one
            if q > self.max_q_ or q_mapping_iterator is None:
                print("Profile " + self.name_ + " is not sampled for q = " + str(q) +
                        ", q_max = " + str(self.max_q_) +
                        "\nYou can remove points with q > " + str(self.max_q_) +
                        " from the experimental profile or recompute the profile with higher max_q")
                return

            i = q_mapping_iterator[1]
            delta_q = 1.0

            if i == 0 or (delta_q := self.q_[i] - self.q_[i - 1]) <= 1.0e-16:
                if hasattr(self, "partial_profiles_") and len(self.partial_profiles_) > 0:
                    for r, pp in enumerate(self.partial_profiles_):
                        resampled_profile.partial_profiles_[r][k] = pp[i]
                resampled_profile.q_[k] = q
                resampled_profile.intensity_[k] = self.intensity_[i]
            else:
                # Interpolate
                alpha = (q - self.q_[i - 1]) / delta_q
                alpha = min(alpha, 1.0) # Handle rounding errors
                if hasattr(self, "partial_profiles_") and len(self.partial_profiles_) > 0:
                    for r, pp in enumerate(self.partial_profiles_):
                        intensity = (1 - alpha) * pp[i - 1] + alpha * pp[i]
                        resampled_profile.partial_profiles_[r][k] = intensity
                intensity = (1 - alpha) * self.intensity_[i - 1] + alpha * self.intensity_[i]
                resampled_profile.q_[k] = q
                resampled_profile.intensity_[k] = intensity

    def calculate_profile_symmetric(self, particles, n, ff_type):
        assert n > 1, f"Attempting to use symmetric computation, symmetry order should be > 1. Got: {n}"
        print(f"start real profile calculation for {len(particles)} particles with symmetry = {n}\n")

        # split units, only number_of_distances units is needed
        number_of_distances = n // 2
        unit_size = len(particles) // n

        # coordinates
        units = []
        for i in range(number_of_distances + 1):
            unit = []
            for j in range(unit_size):
                unit.append(particles[i * unit_size + j].coordinates)
            units.append(unit)

        form_factors = [self.ff_table_.get_form_factor(particle, ff_type) for particle in particles[:unit_size]]

        r_dist = RadialDistributionFunction()

        # distribution within unit
        for i in range(unit_size):
            for j in range(i + 1, unit_size):
                dist2 = get_squared_distance(units[0][i], units[0][j])
                r_dist.add_to_distribution(dist2, 2 * form_factors[i] * form_factors[j])
            r_dist.add_to_distribution(0.0, form_factors[i] * form_factors[i])

        # distributions between units separated by distance i
        for in_ in range(1, number_of_distances):
            for i in range(unit_size):
                for j in range(unit_size):
                    dist2 = get_squared_distance(units[0][i], units[in_][j])
                    r_dist.add_to_distribution(dist2, 2 * form_factors[i] * form_factors[j])

        r_dist.scale(n)

        # distribution between units separated by distance n/2
        r_dist2 = RadialDistributionFunction()
        for i in range(unit_size):
            for j in range(unit_size):
                dist2 = get_squared_distance(units[0][i], units[number_of_distances][j])
                r_dist2.add_to_distribution(dist2, 2 * form_factors[i] * form_factors[j])

        # if n is even, the scale is by n/2
        # if n is odd, the scale is by n
        if n % 2:
            r_dist2.scale(n)  # odd
        else:
            r_dist2.scale(n // 2)  # even

        r_dist2.add(r_dist)

        self.squared_distribution_2_profile(r_dist2)

    def distribution_2_profile(self, r_dist):
        self.init()
        # Iterate over intensity profile
        for k in range(self.size()):
            intensity_k = 0.0

            # Iterate over radial distribution
            for r in range(r_dist.size()):
                dist = get_distance_from_index(r_dist.bin_size, r)
                x = dist * self.q_[k]
                x = math.sin(x)/x if x != 0 else 1
                intensity_k += r_dist[r] * x

            self.intensity_[k] = intensity_k

    def squared_distribution_2_profile(self, r_dist):
        self.init()

        distances = np.sqrt(np.arange(r_dist.size())*r_dist.bin_size)

        use_beam_profile = False
        if self.beam_profile_ is not None and len(self.beam_profile_) > 0:
            use_beam_profile = True

        # # Iterate over intensity profile
        for k in range(self.size()):
            if not use_beam_profile:
                x = np.sinc(distances*self.q_[k]/np.pi)

            else:
                # Needs testing
                x = np.zeros_like(distances)
                for t in range(self.beam_profile_.size()):
                    x1 = np.sinc(distances*math.sqrt(self.q_[k]**2+self.beam_profile_.q_[t]**2))
                    x += 2*self.beam_profile_.intensity_[t]*x1

            intensity_k = np.sum(r_dist.distribution*x)
            self.intensity_[k] = intensity_k

        # For whatever reason, the for loop is faster than the fully vectorized version
        # inn_x = np.multiply.outer(self.q_, distances)
        # x = np.sinc(inn_x/np.pi)
        # intensity = np.sum(x*r_dist.distribution, axis=1)
        # self.intensity_ = intensity

        # Correction for the form factor approximation
        self.intensity_ *= np.exp(-self.modulation_function_parameter_*np.square(self.q_))

    def old_squared_distribution_2_profile(self, r_dist):
        self.init()
        # Precomputed sin(x)/x function
        sf = SincFunction(
            math.sqrt(r_dist.max_distance_) * self.max_q_, 0.0001)

        # Precompute square roots of distances
        distances = [0.0] * r_dist.size()
        for r in range(r_dist.size()):
            if r_dist.distribution[r] != 0.0:
                distances[r] = math.sqrt(get_distance_from_index(r_dist.bin_size, r))

        use_beam_profile = False
        if self.beam_profile_ is not None and len(self.beam_profile_) > 0:
            use_beam_profile = True

        # Iterate over intensity profile
        for k in range(self.size()):
            intensity_k = 0.0

            # Iterate over radial distribution
            for r in range(r_dist.size()):
                if r_dist.distribution[r] != 0.0:
                    dist = distances[r]
                    x = 0.0

                    if use_beam_profile:
                        # Iterate over beam profile
                        for t in range(self.beam_profile_.size()):
                            # x = 2*I(t)*sinc(sqrt(q^2+t^2))
                            x1 = dist * math.sqrt(self.q_[k]**2 + self.beam_profile_.q_[t]**2)
                            # s = math.sin(x1)/x1 if x1 != 0 else 1
                            x += 2 * self.beam_profile_.intensity_[t] * sf.sinc(x1)
                    else:
                        # x = sin(dq)/dq
                        x = dist * self.q_[k]
                        # x = math.sin(x)/x if x != 0 else 1
                        x = sf.sinc(x)

                    # Multiply by the value from distribution
                    intensity_k += r_dist.distribution[r] * x

            # Correction for the form factor approximation
            intensity_k *= math.exp(-self.modulation_function_parameter_ * self.q_[k]**2)
            self.intensity_[k] = intensity_k

    def squared_distributions_2_partial_profiles(self, r_dist):
        r_size = len(r_dist)
        self.init(self.size(), r_size) # reset=False

        distances = np.sqrt(np.arange(r_dist[0].size())*r_dist[0].bin_size)

        use_beam_profile = False
        if self.beam_profile_ is not None and self.beam_profile_.size() > 0:
            use_beam_profile = True


        for k in range(self.size()):

            if not use_beam_profile:
                x = np.sinc(distances*self.q_[k]/np.pi)

            else:
                # Needs testing
                x = np.zeros_like(distances)
                for t in range(self.beam_profile_.size()):
                    x1 = np.sinc(distances*math.sqrt(self.q_[k]**2+self.beam_profile_.q_[t]**2))
                    x += 2*self.beam_profile_.intensity_[t]*x1

            for i in range(r_size):
                intensity_k = np.sum(r_dist[i].distribution*x)
                self.partial_profiles_[i][k] += intensity_k

        corr = np.exp(-self.modulation_function_parameter_*np.square(self.q_))

        for i in range(r_size):
            self.partial_profiles_[i] *= corr

    def old_squared_distributions_2_partial_profiles(self, r_dist):
        r_size = len(r_dist)
        self.init(self.size(), r_size) # reset=False

        sf = SincFunction(math.sqrt(r_dist[0].max_distance_) * self.max_q_, 0.0001)
        distances = np.zeros(r_dist[0].size())
        non_zero_indices = np.where(r_dist[0].distribution > 0.0)[0]
        distances[non_zero_indices] = np.sqrt(get_distance_from_index(r_dist[0].bin_size, non_zero_indices))

        use_beam_profile = False
        if self.beam_profile_ is not None and self.beam_profile_.size() > 0:
            use_beam_profile = True
        for k, f in enumerate(self.q_):
            for r in range(r_dist[0].size()):
                if r_dist[0].distribution[r] > 0.0:
                    dist = distances[r]

                    if use_beam_profile:
                        x1 = dist * np.sqrt(f*f + self.beam_profile_.q_**2)
                        x = np.sum(2 * self.beam_profile_.intensity_ * sf.sinc(x1))
                    else:
                        x = dist * f
                        x = sf.sinc(x)

                    for i in range(r_size):
                        # WARNING: values here seem to be the same between C++ and Python but
                        # with a precision difference, which creates after all the sums a bigger
                        # difference
                        self.partial_profiles_[i][k] += r_dist[i].distribution[r] * x

            corr = math.exp(-self.modulation_function_parameter_ * f * f)
            for i in range(r_size):
                self.partial_profiles_[i][k] *= corr

    def profile_2_distribution(self, rd, max_distance):
        scale = 1.0 / (2 * math.pi * math.pi)
        distribution_size = get_index_from_distance(rd.bin_size, max_distance) + 1

        # Offset profile so that the minimal i(q) is zero
        min_value = self.intensity_[0]
        for k in range(self.size()):
            if self.intensity_[k] < min_value:
                min_value = self.intensity_[k]
        p = Profile(qmin=self.min_q_, qmax=self.max_q_, delta=self.delta_q_, constructor=0)
        p.init()
        for k in range(self.size()):
            p.intensity_[k] = self.intensity_[k] - min_value

        # Iterate over r
        for i in range(distribution_size):
            r = get_distance_from_index(rd.bin_size, i)
            s = 0.0
            # Sum over q: SUM (I(q)*q*sin(qr))
            for k in range(p.size()):
                s += p.intensity_[k] * p.q_[k] * math.sin(p.q_[k] * r)
            rd.add_to_distribution(r, r * scale * s)

    def calculate_profile_reciprocal(self, particles, ff_type):
        if ff_type == FormFactorType.CA_ATOMS:
            print("Reciprocal space profile calculation is not supported for residue level")
            return

        print("Start reciprocal profile calculation for", len(particles), "particles")
        self.init()
        coordinates = [particle.coordinates for particle in particles]
        form_factors = [self.ff_table_.get_form_factors(particle, ff_type) for particle in particles]
        # Iterate over pairs of atoms
        for i in range(len(coordinates)):
            factor_i = form_factors[i]
            for j in range(i + 1, len(coordinates)):
                factor_j = form_factors[j]
                dist = get_distance(coordinates[i], coordinates[j])
                for k in range(self.size()):
                    x = dist * self.q_[k]
                    x = math.sin(math.pi * x) / (math.pi * x)
                    self.intensity_[k] += 2 * x * factor_i * factor_j

            # Add autocorrelation part
            for k in range(self.size()):
                self.intensity_[k] += factor_i* factor_i

    def calculate_profile_reciprocal_partial(self, particles, surface, ff_type):
        if ff_type == FormFactorType.CA_ATOMS:
            print("Reciprocal space profile calculation is not supported for residue level")
            return

        print("Start partial reciprocal profile calculation for", len(particles), "particles")

        coordinates = [particle.coordinates for particle in particles]

        r_size = 3
        if len(surface) == len(particles):
            r_size = 6
        water_ff = self.ff_table_.get_water_form_factors()
        self.init(0, r_size)

        for i in range(len(coordinates)):
            vacuum_ff1 = self.ff_table_.get_vacuum_form_factors(particles[i], ff_type)
            dummy_ff1 = self.ff_table_.get_dummy_form_factors(particles[i], ff_type)
            for j in range(i + 1, len(coordinates)):
                vacuum_ff2 = self.ff_table_.get_vacuum_form_factors(particles[j], ff_type)
                dummy_ff2 = self.ff_table_.get_dummy_form_factors(particles[j], ff_type)
                dist = get_distance(coordinates[i], coordinates[j])

                for k in range(self.size()):
                    x = dist * self.q_[k]
                    x = 2 * math.sin(math.pi * x) / (math.pi * x)
                    self.partial_profiles_[0][k] += x * vacuum_ff1[k] * vacuum_ff2[k]
                    self.partial_profiles_[1][k] += x * dummy_ff1[k] * dummy_ff2[k]
                    self.partial_profiles_[2][k] += x * (vacuum_ff1[k] * dummy_ff2[k] +
                                                        vacuum_ff2[k] * dummy_ff1[k])

                    if r_size > 3:
                        self.partial_profiles_[3][k] += x * surface[i] * surface[j] * water_ff[k] * water_ff[k]
                        self.partial_profiles_[4][k] += x * (vacuum_ff1[k] * surface[j] * water_ff[k] +
                                                            vacuum_ff2[k] * surface[i] * water_ff[k])
                        self.partial_profiles_[5][k] += x * (dummy_ff1[k] * surface[j] * water_ff[k] +
                                                            dummy_ff2[k] * surface[i] * water_ff[k])

            for k in range(self.size()):
                self.partial_profiles_[0][k] += vacuum_ff1[k] * vacuum_ff1[k]
                self.partial_profiles_[1][k] += dummy_ff1[k] * dummy_ff1[k]
                self.partial_profiles_[2][k] += 2 * vacuum_ff1[k] * dummy_ff1[k]

                if r_size > 3:
                    self.partial_profiles_[3][k] += surface[i] * surface[i] * water_ff[k] * water_ff[k]
                    self.partial_profiles_[4][k] += 2 * vacuum_ff1[k] * surface[i] * water_ff[k]
                    self.partial_profiles_[5][k] += 2 * dummy_ff1[k] * surface[i] * water_ff[k]

        self.sum_partial_profiles(1.0, 0.0, False)

    def calculate_profile(self, particles, ff_type=FormFactorType.HEAVY_ATOMS, gpu=False):
        if not gpu:
            self.calculate_profile_real(particles, ff_type)
        else:
            self.calculate_profile_real_gpu(particles, ff_type)

    def size(self):
        return len(self.q_) if hasattr(self, "q_") else 0

class FitParameters:
    def __init__(self, chi_square=0.0, c1=0.0, c2=0.0,
        c=0.0, o=0.0, default_chi_square=0.0):
        self.chi_square = chi_square
        self.c1 = c1
        self.c2 = c2
        self.c = c
        self.o = o
        self.default_chi_square = default_chi_square

    def __lt__(self, other):
        return self.chi_square < other.chi_square

class ProfileFitter:
    def __init__(self, exp_profile):
        self.exp_profile_ = exp_profile
        self.scoring_function_ = ChiScore()

    def compute_scale_factor(self, model_profile, offset=0.0):
        return self.scoring_function_.compute_scale_factor(self.exp_profile_, model_profile, offset)

    def compute_offset(self, model_profile):
        return self.scoring_function_.compute_offset(self.exp_profile_, model_profile)

    def get_profile(self):
        return self.exp_profile_

    def search_fit_parameters(self, partial_profile, min_c1, max_c1, min_c2,
        max_c2, use_offset, old_chi):
        c1_cells = 10
        c2_cells = 10
        if old_chi < float('inf') - 1:  # second iteration
            c1_cells = 5
            c2_cells = 5

        delta_c1 = (max_c1 - min_c1) / c1_cells
        delta_c2 = (max_c2 - min_c2) / c2_cells

        last_c1 = False
        last_c2 = False
        if delta_c1 < 0.0001:
            c1_cells = 1
            delta_c1 = max_c1 - min_c1
            last_c1 = True
        if delta_c2 < 0.001:
            c2_cells = 1
            delta_c2 = max_c2 - min_c2
            last_c2 = True

        best_c1 = 1.0
        best_c2 = 0.0
        best_chi = old_chi
        best_set = False

        c1 = min_c1
        for _ in range(c1_cells + 1):
            c2 = min_c2
            for _ in range(c2_cells + 1):
                partial_profile.sum_partial_profiles(c1, c2)
                curr_chi, fit_profile = self.compute_score(partial_profile, use_offset)
                if not best_set or curr_chi < best_chi:
                    best_set = True
                    best_chi = curr_chi
                    best_c1 = c1
                    best_c2 = c2
                c2 += delta_c2
            c1 += delta_c1

        if abs(best_chi - old_chi) > 0.0001 and not (last_c1 and last_c2):
            min_c1 = max(best_c1 - delta_c1, min_c1)
            max_c1 = min(best_c1 + delta_c1, max_c1)
            min_c2 = max(best_c2 - delta_c2, min_c2)
            max_c2 = min(best_c2 + delta_c2, max_c2)
            return self.search_fit_parameters(partial_profile, min_c1, max_c1, min_c2, max_c2, use_offset, best_chi)
        return FitParameters(best_chi, best_c1, best_c2)

    def fit_profile(self, partial_profile, min_c1, max_c1, min_c2, max_c2, use_offset):
        # Compute chi value for default c1/c2
        default_c1 = 1.0
        default_c2 = 0.0
        partial_profile.sum_partial_profiles(default_c1, default_c2)
        default_chi, fit_profile = self.compute_score(partial_profile, use_offset)

        fp = self.search_fit_parameters(partial_profile, min_c1, max_c1, min_c2, max_c2, use_offset, float('inf'))
        best_c1 = fp.c1
        best_c2 = fp.c2
        fp.default_chi_square = default_chi
        # Compute a profile for the best c1/c2 combination
        partial_profile.sum_partial_profiles(best_c1, best_c2)
        score, fit_profile = self.compute_score(partial_profile, use_offset)
        return fit_profile, score, fp

    def compute_score(self, model_profile, use_offset):
        resampled_profile = Profile(
            qmin=self.exp_profile_.min_q_,
            qmax=self.exp_profile_.max_q_,
            delta=self.exp_profile_.delta_q_,
            constructor=0
        )
        # model_profile and resampled_profile might be different than the C++ version
        model_profile.resample(self.exp_profile_, resampled_profile)
        score = self.scoring_function_.compute_score(self.exp_profile_, resampled_profile, use_offset)

        offset = 0.0
        if use_offset:
            offset = self.scoring_function_.compute_offset(self.exp_profile_, resampled_profile)
        c = self.scoring_function_.compute_scale_factor(self.exp_profile_, resampled_profile, offset)

        resampled_profile.intensity_ = resampled_profile.intensity_ * c - offset

        return score, resampled_profile

def get_distance(vector1, vector2):
    # Convert the vectors to NumPy arrays
    array1 = np.array(vector1)
    array2 = np.array(vector2)

    # Calculate the distance
    dist = np.sqrt(np.sum((array1 - array2) ** 2))

    return dist

def get_squared_distance(vector1, vector2):
    # Convert the vectors to NumPy arrays
    array1 = np.array(vector1)
    array2 = np.array(vector2)

    # Calculate the squared distance
    squared_dist = np.sum((array1 - array2) ** 2)

    return squared_dist

def inner_resample(q_rs, intensity_rs, pp_rs, exp_q, q_, intensity_, max_q_,
    name_, pp_):
    # Initialize
    size_pp = len(pp_)
    size_q = len(q_)

    next_q_idx = 0

    for k in range(len(exp_q)):
        q = exp_q[k]

        found_next = False
        while not found_next:
            if q_[next_q_idx] >= q:
                found_next = True
            elif next_q_idx ==  size_q -1:
                break
            else:
                next_q_idx += 1

        # In case the experimental profile is longer than the computed one
        if q > max_q_ or not found_next:
            print("The experimental profile is sample to a higher q than the "
                "set maximum q for the theoretical profile. In order to fit the "
                "profile, you must either trim the experimental profile to the "
                "appropriate maximum q value or set the maximum q value of the "
                "theoretical profile to be generated to be greater than that of "
                "the experimental profile.")
            return

        i = next_q_idx
        delta_q = q_[i] - q_[i - 1]

        if i == 0 or delta_q <= 1.0e-16:
            if size_pp > 0:
                for r, pp in enumerate(pp_):
                    pp_rs[r][k] = pp[i]
            q_rs[k] = q
            intensity_rs[k] = intensity_[i]
        else:
            # Interpolate
            alpha = (q - q_[i - 1]) / delta_q
            alpha = min(alpha, 1.0) # Handle rounding errors
            if size_pp > 0:
                for r, pp in enumerate(pp_):
                    intensity = (1 - alpha) * pp[i - 1] + alpha * pp[i]
                    pp_rs[r][k] = intensity
            intensity = (1 - alpha) * intensity_[i - 1] + alpha * intensity_[i]
            q_rs[k] = q
            intensity_rs[k] = intensity

    return q_rs, intensity_rs, pp_rs

def inner_calculate_profile_real(coordinates, form_factors, one_over_bin_size):
    # iterate over pairs of atoms
    distribution = np.zeros(1)

    for i in range(len(coordinates)):
        dists = np.sum(np.square(coordinates[i+1:] - coordinates[i]),axis=1)
        prods = 2*form_factors[i+1:]*form_factors[i]

        if dists.size > 0:
            max_val = int(dists.max()*one_over_bin_size+0.5)
            if max_val > distribution.size:
                ext = np.zeros(max_val-distribution.size+1)
                distribution = np.concatenate((distribution, ext))

            for k in range(dists.size):
                distribution[int(dists[k]*one_over_bin_size+0.5)] += prods[k]

    distribution[0] += np.sum(np.square(form_factors))

    return distribution

def inner_calculate_profile_partial(coordinates, vacuum_ff, dummy_ff, water_ff,
    r_size, one_over_bin_size):
    # iterate over pairs of atoms
    distributions = [np.zeros(1) for i in range(r_size)]

    for i in range(len(coordinates)):
        dists = np.sum(np.square(coordinates[i+1:] - coordinates[i]),axis=1)

        prod1 = 2*vacuum_ff[i+1:]*vacuum_ff[i] # constant
        prod2 = 2*dummy_ff[i+1:]*dummy_ff[i] # c1^2
        prod3 = 2*(vacuum_ff[i+1:]*dummy_ff[i] + dummy_ff[i+1:]*vacuum_ff[i]) # -c1

        if r_size > 3:
            prod4 = 2*water_ff[i+1:]*water_ff[i] # c2^2
            prod5 = 2*(vacuum_ff[i+1:]*water_ff[i] + water_ff[i+1:]*vacuum_ff[i]) # c2
            prod6 = 2*(dummy_ff[i+1:]*water_ff[i] + water_ff[i+1:]*dummy_ff[i]) # -c1*c2

        if dists.size > 0:
            max_val = int(dists.max()*one_over_bin_size+0.5)
            if max_val > distributions[0].size:
                for j in range(r_size):
                    ext = np.zeros(max_val-distributions[j].size+1)
                    distributions[j] = np.concatenate((distributions[j], ext))

            for k in range(dists.size):
                index = int(dists[k]*one_over_bin_size+0.5)
                distributions[0][index] += prod1[k]
                distributions[1][index] += prod2[k]
                distributions[2][index] += prod3[k]

                if r_size > 3:
                    distributions[3][index] += prod4[k]
                    distributions[4][index] += prod5[k]
                    distributions[5][index] += prod6[k]

    distributions[0][0] += np.sum(np.square(vacuum_ff))
    distributions[1][0] += np.sum(np.square(dummy_ff))
    distributions[2][0] += 2*np.sum(vacuum_ff*dummy_ff)

    if r_size > 3:
        distributions[3][0] += np.sum(np.square(water_ff))
        distributions[4][0] += 2*np.sum(water_ff*vacuum_ff)
        distributions[5][0] += 2*np.sum(water_ff*dummy_ff)

    return distributions

def inner_calculate_profile_real_gpu(coordinates, form_factors, one_over_bin_size):
    # iterate over pairs of atoms
    # We move our tensor to the GPU if available
    if torch.cuda.is_available():
        coordinates = coordinates.to("cuda")
        form_factors = form_factors.to("cuda")

    # distribution = torch.tensor([torch.sum(torch.square(form_factors))])
    # if torch.cuda.is_available():
    #     distribution = distribution.to("cuda")

    ff0 = torch.sum(torch.square(form_factors))
    ff0 = ff0.cpu()
    ff0 = ff0.numpy()
    distribution = np.array([ff0])

    for i in range(len(coordinates)):
        dists = torch.sum(torch.square(coordinates[i+1:] - coordinates[i]),axis=1)
        prods = torch.multiply(form_factors[i+1:], 2*form_factors[i])

        if torch.numel(dists) > 0:
            dists = dists*one_over_bin_size+0.5
            dists = dists.to(torch.int32)
            # dist_cpu = dists.cpu()
            # prods_cpu = prods.cpu()
            # dist_cpu = dist_cpu.numpy()
            # prods_cpu = prods_cpu.numpy()
            # assign_vals(distribution, dist_cpu, prods_cpu, one_over_bin_size)
            max_val = dists.max()
            dist_size = distribution.size
            if max_val > dist_size:

                # ext = torch.zeros(int(max_val-dist_size+1))
                # if torch.cuda.is_available():
                #     ext = ext.to("cuda")
                # distribution = torch.concatenate((distribution, ext))

                ext = np.zeros(int(max_val-dist_size+1))
                distribution = np.concatenate((distribution, ext))

            # for k in range(torch.numel(dists)):
            #     distribution[dists[k]] += prods[k]

            dists = dists.cpu()
            dists = dists.numpy()
            prods = prods.cpu()
            prods = prods.numpy()

            distribution = set_dist_vals(distribution, dists, prods)

    # distribution_cpu = distribution.cpu()
    # distribution_cpu = distribution_cpu.numpy()
    distribution_cpu = distribution
    return distribution_cpu


def assign_vals(distribution, dists, prods, one_over_bin_size):
    max_val = int(dists.max()*one_over_bin_size+0.5)

    if max_val > distribution.size:
        ext = np.zeros(max_val-distribution.size+1)
        distribution = np.concatenate((distribution, ext))

    for k in range(dists.size):
        distribution[int(dists[k]*one_over_bin_size+0.5)] += prods[k]

def set_dist_vals(distribution, dists, prods):
    for k in range(dists.size):
        distribution[dists[k]] += prods[k]

    return distribution

# def inner_calculate_profile_real_gpu(coordinates, form_factors, one_over_bin_size):
#     pass


# test_kernel = cp.ElementwiseKernel(
#     'raw T coord, raw T ff',
#     'raw T dists, raw T prods',
#     '''

#     ''',
#     'test_kernel'
#     )

def test_mult(x, mult):
    return np.multiply(x, mult)


def compute_profile(particles, min_q, max_q, delta_q, ff_type):
    profile = Profile(qmin=min_q, qmax=max_q, delta=delta_q, constructor=0)
    ft = profile.ff_table_
    surface_area = []
    s = SolventAccessibleSurface()
    average_radius = 0.0
    for particle in particles:
        radius = ft.get_radius(particle, ff_type)
        particle.radius = radius
        average_radius += radius
    surface_area = s.get_solvent_accessibility(particles)
    average_radius /= len(particles)
    profile.average_radius_ = average_radius

    profile.calculate_profile_partial(particles, surface_area, ff_type)
    return profile

def fit_profile(exp_profile, model_profile, min_c1, max_c1, min_c2, max_c2, use_offset):
    fitter = ProfileFitter(exp_profile)
    fit_profile, chi_square, fp = fitter.fit_profile(model_profile, min_c1, max_c1, min_c2, max_c2, use_offset)
    return fit_profile, chi_square, fp