"""
\file IMP/saxs/SolventAccessibleSurface.h \brief

Copyright 2007-2022 IMP Inventors. All rights reserved.
"""

import math
import numpy as np
from scipy import spatial
from numba import jit

class SolventAccessibleSurface:
    def __init__(self):
        self.sphere_dots = None

    def compute_accessibility(self, point, probe_radius, density):
        sphere_dots = self.create_sphere_dots(probe_radius, density)
        intersecting_count = 0
        for sphere_center, radius in sphere_dots:
            if is_intersecting(point, sphere_center, point[3], radius):
                intersecting_count += 1
        accessibility = 1.0 - (intersecting_count / len(sphere_dots))
        return accessibility

    def get_solvent_accessibility(self, ps, probe_radius=1.8, density=5.0):
        res = []
        coordinates = np.array([p.coordinates for p in ps])
        radii = np.array([p.radius for p in ps])

        # generate sphere dots for radii present in the ps set
        self.create_sphere_dots(ps, density)

        # init grid
        grid = spatial.KDTree(coordinates)

        # res2 = []

        # compute surface
        max_radius = 3.0
        for i in range(len(ps)):
            atom_radius = radii[i]
            radius = atom_radius + 2 * probe_radius + max_radius
            spoints = self.sphere_dots[atom_radius]

            inside_rad = np.array(grid.query_ball_point(coordinates[i], radius),
                dtype=np.int_)
            # inside_rad = np.setdiff1d(inside_rad, i)

            dotNum = inner_get_solvent_accessibilty(inside_rad, spoints, radii,
                coordinates[i], coordinates, probe_radius, atom_radius)

            res.append(float(dotNum) / len(spoints))


        return res

    def old_get_solvent_accessibility(self, ps, probe_radius=1.8, density=5.0):
        res = []
        coordinates = np.array([p.coordinates for p in ps])
        radii = np.array([p.radius for p in ps])

        # generate sphere dots for radii present in the ps set
        self.create_sphere_dots(ps, density)

        # init grid
        grid = spatial.KDTree(coordinates)

        # compute surface
        max_radius = 3.0
        for i in range(len(ps)):
            atom_radius = radii[i]
            radius = atom_radius + 2 * probe_radius + max_radius

            inside_rad = grid.query_ball_point(coordinates[i], radius)

            neighbours1, neighbours2 = [], []
            for mol_index in inside_rad:
                radius_sum1 = atom_radius + radii[mol_index]
                radius_sum2 = radius_sum1 + 2 * probe_radius
                dist2 = get_squared_distance(coordinates[i], coordinates[mol_index])
                if dist2 < radius_sum1 * radius_sum1:
                    neighbours1.append(mol_index)
                elif dist2 < radius_sum2 * radius_sum2:
                    neighbours2.append(mol_index)

            ratio = (atom_radius + probe_radius) / atom_radius
            spoints = np.array(self.get_sphere_dots(atom_radius))

            dotNum = 0
            for s_index in range(len(spoints)):
                probe_center = coordinates[i] + ratio * spoints[s_index]
                # check for intersection with neighbors1
                collides = False
                for n_index in range(len(neighbours1)):
                    if is_intersecting(probe_center, coordinates[neighbours1[n_index]],
                                            probe_radius, radii[neighbours1[n_index]]):
                        collides = True
                        break
                if not collides: # check for intersection with neighbors2
                    for n_index in range(len(neighbours2)):
                        if is_intersecting(probe_center, coordinates[neighbours2[n_index]],
                                                probe_radius, radii[neighbours2[n_index]]):
                            collides = True
                            break
                if not collides:
                    dotNum += 1
            res.append(float(dotNum) / len(spoints))
        return res

    def get_nearest_index(self, grid_coordinates, target_coordinates):
        # Convert the grid coordinates and target coordinates to NumPy arrays
        grid_coordinates = np.array(grid_coordinates)
        target_coordinates = np.array(target_coordinates)

        # Calculate the Euclidean distances between the target coordinates and all grid coordinates
        distances = np.linalg.norm(grid_coordinates - target_coordinates, axis=1)

        # Find the index corresponding to the minimum distance
        nearest_index = np.argmin(distances)

        return nearest_index

    def create_sphere_dots(self, ps, density):
        sphere_dots = {}

        for p in ps:
            r = p.radius # [3] # radius in XYZR
            if r not in sphere_dots:
                dots = self.create_sphere_dots_for_radius(r, density)
                sphere_dots[r] = np.array(dots)

        self.sphere_dots = sphere_dots

    def get_sphere_dots(self, r):
        return self.sphere_dots[r]

    def create_sphere_dots_for_radius(self, radius, density):
        res = []
        num_equat = 2 * math.pi * radius * math.sqrt(density)
        vert_count = 0.5 * num_equat

        for i in range(math.ceil(vert_count)):
            phi = (math.pi * i) / vert_count
            z = math.cos(phi)
            xy = math.sin(phi)
            horz_count = xy * num_equat
            for j in range(math.ceil(horz_count - 1)):
                teta = (2 * math.pi * j) / horz_count
                x = xy * math.cos(teta)
                y = xy * math.sin(teta)
                res.append((radius * x, radius * y, radius * z))
        return res

@jit(nopython=True, cache=True)
def get_squared_distance(v1, v2):
    squared_dist = 0.0
    for i in range(3):
        squared_dist += (v1[i] - v2[i]) ** 2
    return squared_dist

@jit(nopython=True, cache=True)
def is_intersecting(center1, center2, radius1, radius2):
    squared_radius_sum = (radius1 + radius2) * (radius1 + radius2)
    squared_dist = get_squared_distance(center1, center2)

    if squared_radius_sum - squared_dist > 0.0001:
        return True

    return False

@jit(nopython=True, cache=True)
def inner_get_solvent_accessibilty(inside_rad, spoints, radii, current_coord, coordinates,
    probe_radius, atom_radius):

    radius_sum1 = atom_radius + radii[inside_rad]
    radius_sum2 = radius_sum1+2*probe_radius
    dist2 = np.sum(np.square(coordinates[inside_rad]-current_coord),axis=1)

    neighbours1 = inside_rad[dist2<np.square(radius_sum1)]
    neighbours2 = inside_rad[dist2<np.square(radius_sum2)]

    ratio = (atom_radius + probe_radius) / atom_radius
    dotNum = 0
    tot_points = len(spoints)
    tot_n1 = len(neighbours1)
    tot_n2 = len(neighbours2)

    for s_index in range(tot_points):
        probe_center = current_coord + ratio * spoints[s_index]
        # check for intersection with neighbors1
        collides = False
        for n_index in range(tot_n1):
            n1 = neighbours1[n_index]
            if is_intersecting(probe_center, coordinates[n1], probe_radius,
                radii[n1]):
                collides = True
                break

        if not collides: # check for intersection with neighbors2
            for n_index in range(tot_n2):
                n2 = neighbours2[n_index]
                if is_intersecting(probe_center, coordinates[n2], probe_radius,
                    radii[n2]):
                    collides = True
                    break

        if not collides:
            dotNum += 1

    return dotNum
