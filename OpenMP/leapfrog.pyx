# Cython script for OpenMP-parallelized simulations. Use setup.py to compile.

from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np

cdef int G = 1  # Gravitational constant

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def leapfrog_step(double[:,:] positions, 
                  double[:,:] velocities, 
                  double[:,:] accels, 
                  double[:] masses, 
                  double dt, 
                  double softening,
                  int threads):
    """Perform a kick-drift-kick leapfrog step on all particles. Paralellized with OpenMP multithreading.

    Args:
        positions (NDArray): positions of all particles
        velocities (NDArray): velocities of all particles
        accels (NDArray): accelerations of all particles
        masses (NDArray): masses of all particles 
        dt (float): time step size
        softening_length (float): epsilon - softening parameter for gravitational interactions
        threads (int): number of threads to distribute workload at loop-level

    Returns:
        tuple: (updated) positions, velocities, accels (NDArray objects)
    """
    cdef:
        int N = positions.shape[0]
        int i

    # Kick: update velocities by half timestep
    for i in prange(N,nogil=True,num_threads=threads):
        velocities[i, 0] += 0.5 * dt * accels[i,0]
        velocities[i, 1] += 0.5 * dt * accels[i,1]
        velocities[i, 2] += 0.5 * dt * accels[i,2]

    # Drift: update positions by full timestep
    for i in prange(N,nogil=True,num_threads=threads):
        positions[i, 0] += dt * velocities[i, 0]
        positions[i, 1] += dt * velocities[i, 1]
        positions[i, 2] += dt * velocities[i, 2]

    # Recalculate accels
    accels = calculate_accels(positions, masses, softening, threads)

    # Kick: update velocities by another half timestep
    for i in prange(N,nogil=True,num_threads=threads):
        velocities[i, 0] += 0.5 * dt * accels[i,0]
        velocities[i, 1] += 0.5 * dt * accels[i,1]
        velocities[i, 2] += 0.5 * dt * accels[i,2]
    
    return positions, velocities, accels


@cython.boundscheck(False)  # Disable bounds checking for performance
@cython.wraparound(False)   # Disable negative indexing
def calculate_accels(double[:, :] positions, 
                     double[:] masses,
                     double softening,
                     int threads):
    """
    Compute acceleration due to gravity on all particles due to all other particles. Parallelized using OpenMP.
    """
    cdef:
        int N = positions.shape[0]
        int i, j
        double dx, dy, dz, softened_r2,softened_r, inv_r3
        double[:, :] accels = np.zeros_like(positions)  # Create the accel array
    
    # Parallel loop for force calculation
    for i in prange(N, nogil=True, schedule='static',num_threads=threads):
        for j in range(N):
            if i != j :
                # r_vec:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]

                # r squared
                softened_r2 = dx * dx + dy * dy + dz * dz + (softening * softening)
                softened_r = softened_r2**0.5
                inv_r3 = (softened_r)**(-3)  # 1/r^3 then to be multiplied by (dx,dy,dz). Here we have substituted r_hat for r/mag_r
                
                accels[i, 0] += G * masses[j] * inv_r3 * dx
                accels[i, 1] += G * masses[j] * inv_r3 * dy
                accels[i, 2] += G * masses[j] * inv_r3 * dz

    return(accels)