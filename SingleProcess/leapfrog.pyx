# Cython script for serial simulations. Use setup.py to compile.

import cython
import numpy as np
cimport numpy as np

cdef int G = 1

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def leapfrog_step(double[:, :] positions,double[:, :] velocities,double[:,:] accels,double[:] masses,double dt, double softening_length):
    """Perform a kick-drift-kick leapfrog step on all particles.

    Args:
        positions (NDArray): positions of all particles
        velocities (NDArray): velocities of all particles
        accels (NDArray): accelerations of all particles
        masses (NDArray): masses of all particles 
        dt (float): time step size
        softening_length (float): epsilon - softening parameter for gravitational interactions

    Returns:
        tuple: (updated) positions, velocities, accels (NDArray objects)
    """
    
    cdef:
        int i
        
    # Compute the velocity update: velocities
    for i in range(velocities.shape[0]):  # Loop over each particle
        velocities[i, 0] += 0.5 * accels[i, 0] * dt
        velocities[i, 1] += 0.5 * accels[i, 1] * dt
        velocities[i, 2] += 0.5 * accels[i, 2] * dt

    # Compute the position update
    for i in range(positions.shape[0]):  # Loop over each particle
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt

    # Recalculate accelerations after position update
    accels = calculate_accels(positions, masses, softening_length)

    # Compute the second velocity update
    for i in range(velocities.shape[0]):  # Loop over each particle
        velocities[i, 0] += 0.5 * accels[i, 0] * dt
        velocities[i, 1] += 0.5 * accels[i, 1] * dt
        velocities[i, 2] += 0.5 * accels[i, 2] * dt
    
    return positions, velocities, accels

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def calculate_accels(double[:, :] positions,double[:] masses, double softening_length):
    """
    Compute acceleration due to gravity on all particles due to all other particles.
    """
    cdef:
        int num_particles = masses.shape[0]
        double[:, :] accels = np.zeros_like(positions)  # Create the accel array
        double[:] r_vec = np.empty(3)  # Temporary vector for calculations
        double softened_r, accel
        int i, j, k
    
    for i in range(num_particles):
        for j in range(num_particles):
            if i!=j:    # If different particles
                # r_vec:
                for k in range(3):
                    r_vec[k] = positions[j,k] - positions[i,k]

                # r squared
                softened_r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2] + softening_length**2
                softened_r = softened_r2**(0.5)
                accel = G *masses[j] * softened_r**(-3)  # 1/r^3 then to be multiplied by r_hat. Here we have substituted r_hat for r_vec/mag_r

                for k in range(3):
                    accels[i, k] += accel * r_vec[k]

    return accels