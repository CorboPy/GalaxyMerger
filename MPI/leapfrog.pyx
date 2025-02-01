# Cython script for MPI functions. OpenMP is incorporated here for hybrid OpenMP + MPI jobs. Use setup.py to compile.

from cython.parallel cimport prange
import cython
import numpy as np
cimport numpy as np

cdef int G = 1  # gravitational constant

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def leapfrog_step(double[:,:] global_positions, 
                  double[:] global_masses, 
                  double[:,:] local_accels, 
                  double[:,:] local_velocities,
                  int[:] local_indicies, 
                  double dt, 
                  double softening_length,
                  int threads):
    """Perform a kick-drift-kick leapfrog step on local_particles given global_positions. Paralellized with OpenMP multithreading.

    Args:
        global_positions (NDArray): positions of all particles across all ranks
        global_masses (NDArray): masses of all particles across all ranks
        local_accels (NDArray): accelerations of particles local to a rank
        local_velocities (NDArray): velocities of particles local to a rank
        local_indicies (NDArray): indicies of particles local to a rank (used for indexing global_positions)
        dt (float): time step size
        softening_length (float): epsilon - softening parameter for gravitational interactions
        threads (int): number of threads to distribute workload at loop-level

    Returns:
        tuple: (updated) local_positions, local_velocities, local_accels (NDArray objects)
    """
    cdef:
        int num_local_particles = local_indicies.shape[0]
        double[:,:] local_positions = np.zeros((num_local_particles, 3))  # assuming 3D positions.
        int i

    # Manually index based on local_indicies
    for i in range(num_local_particles):
        local_positions[i, 0] = global_positions[local_indicies[i], 0]
        local_positions[i, 1] = global_positions[local_indicies[i], 1]
        local_positions[i, 2] = global_positions[local_indicies[i], 2]
        
    # Compute the velocity update: 
    for i in prange(num_local_particles,nogil=True,num_threads=threads):  # Loop over each particle
        local_velocities[i, 0] += 0.5 * local_accels[i,0] * dt
        local_velocities[i, 1] += 0.5 * local_accels[i,1] * dt
        local_velocities[i, 2] += 0.5 * local_accels[i,2] * dt

    # Compute the position update
    for i in prange(num_local_particles,nogil=True,num_threads=threads):  # Loop over each particle
        local_positions[i, 0] += local_velocities[i, 0] * dt
        local_positions[i, 1] += local_velocities[i, 1] * dt
        local_positions[i, 2] += local_velocities[i, 2] * dt

    # Recalculate accels after position update
    local_accels = calculate_local_accels(local_positions, local_indicies, global_positions, global_masses, softening_length,threads)

    # Compute the second velocity update
    for i in prange(num_local_particles,nogil=True,num_threads=threads):  # Loop over each particle
        local_velocities[i, 0] += 0.5 * local_accels[i,0] * dt
        local_velocities[i, 1] += 0.5 * local_accels[i,1] * dt
        local_velocities[i, 2] += 0.5 * local_accels[i,2] * dt

    return local_positions, local_velocities, local_accels


@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def calculate_local_accels(double[:,:] local_positions, 
                           int[:] local_indicies,
                           double[:,:] global_positions, 
                           double[:] global_masses, 
                           double softening_length,
                           int threads):
    """
    Compute acceleration due to gravity on local particles due to global particles. Parallelized using OpenMP.
    """
    cdef:
        double[:,:] local_accels = np.zeros_like(local_positions)
        int num_particles = global_masses.shape[0]
        int i, j
        int particle_index
        double dx, dy, dz, softened_r2, softened_r, inv_r3
        int local_size = local_indicies.shape[0]
    
    for i in prange(local_size,schedule='static',nogil=True,num_threads=threads):
        # i = 0,1,2,3... N_local-1
        particle_index = local_indicies[i]
        # particle_index = the GLOBAL index of the local particle
        for j in range(num_particles):
            # j = 0,1,2,3 ... N-1
            if particle_index != j:    # If they are different particles

                # r_vec:
                dx = global_positions[j, 0] - local_positions[i, 0]
                dy = global_positions[j, 1] - local_positions[i, 1]
                dz = global_positions[j, 2] - local_positions[i, 2]            
   
                # r squared
                softened_r2 = dx * dx + dy * dy + dz * dz + (softening_length * softening_length)
                softened_r = softened_r2**0.5
                inv_r3 = (softened_r)**(-3)  # 1/r^3 then to be multiplied by (dx,dy,dz). Here we have substituted r_hat for r/mag_r

                local_accels[i, 0] += G * global_masses[j] * inv_r3 * dx
                local_accels[i, 1] += G * global_masses[j] * inv_r3 * dy
                local_accels[i, 2] += G * global_masses[j] * inv_r3 * dz
                
    return local_accels