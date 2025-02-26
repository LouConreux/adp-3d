"""
\file ChiScore.h   \brief Basic SAXS scoring

Copyright 2007-2022 IMP Inventors. All rights reserved.
"""

import numpy as np

class ChiScore:
    def compute_score(self, exp_profile, model_profile, use_offset):
        offset = 0.0
        if use_offset:
            offset = self.compute_offset(exp_profile, model_profile)
        c = self.compute_scale_factor(exp_profile, model_profile, offset)

        chi_square = 0.0
        profile_size = min(model_profile.size(), exp_profile.size())

        errors = exp_profile.error_
        exp_intensities = exp_profile.intensity_
        model_intensities = model_profile.intensity_

        delta = exp_intensities - c * model_intensities
        if use_offset:
            delta += offset

        chi_square = np.sum(np.square(delta/errors))

        chi_square /= profile_size

        return chi_square

    def compute_scale_factor(self, exp_profile, model_profile, offset):
        errors = exp_profile.error_
        exp_intensities = exp_profile.intensity_
        model_intensities = model_profile.intensity_

        square_errors = np.square(errors)
        square_errors = 1.0 / square_errors

        sum_imod2 = np.sum(square_errors * model_intensities * model_intensities)
        sum_imod_iexp = np.sum(square_errors * model_intensities * exp_intensities)
        c = sum_imod_iexp / sum_imod2

        if np.fabs(offset) > 1.0e-10:
            sum_imod = np.sum(square_errors * model_intensities)
            constant = sum_imod / sum_imod2
            c += offset * constant

        return c

    def compute_offset(self, exp_profile, model_profile):
        errors = exp_profile.error_
        exp_intensities = exp_profile.intensity_
        model_intensities = model_profile.intensity_

        square_errors = np.square(errors)
        square_errors = 1.0 / square_errors

        sum_imod = np.sum(square_errors * model_intensities)
        sum_imod2 = np.sum(square_errors * (model_intensities * model_intensities))
        constant = sum_imod / sum_imod2

        sum_imod_iexp = np.sum(square_errors * model_intensities * exp_intensities)
        c = sum_imod_iexp / sum_imod2

        delta = exp_intensities - c * model_intensities
        delta2 = constant * model_intensities
        delta2 = 1.0 - delta2

        sum1 = np.sum(square_errors * delta * delta2)
        sum2 = np.sum(square_errors * delta2 * delta2)
        offset = -sum1 / sum2

        return offset
