## make_galaxy.py creates an exponential disk of N particles in simulation units (G=1) and plots it, outputting plots for (x,y), (y,z), (x,y,z), (r, v_c) 

# How to use:
#   - adjust galaxy parameters as needed (line 66)
#   - run this script adding integer as an argument for N. e.g., to create a galaxy with 10,000 particles, do: python make_galaxy.py 10000
#   - view plots created (IC_etc.png) to verify 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
import sys

G_si = 6.67430*10**-11  # Gravitational constant in m^3 kg^-1 s^-2
G=1 # Sim units

def circular_velocity(r, M_enclosed):
    """Calculate circular velocity for stableish orbits given enclosed mass."""
    return np.sqrt(G * M_enclosed / r)

def create_disk(num_particles, mass, scale_radius, z_spread):
    """Creates a 3D exponential disk of bodies

    Args:
        num_particles (float): number of bodies
        mass (float): total mass of the galaxy (=1 in sim units)
        scale_radius (float): exponential scale radius (=1 in sim units)
        z_spread (float): exponential scale height of galaxy (in units of scale_radius)

    Returns:
        tuple: positions, velocities, masses (NDArray)
    """
    pos = np.zeros((num_particles, 3))
    vel = np.zeros((num_particles, 3))
    masses = np.ones(num_particles) * (mass / num_particles)    # Each star has a mass = total mass of galaxy / No. of particles

    for i in range(num_particles):
        r = np.random.exponential(scale=scale_radius)   # Exponential PDF
        phi = 2 * np.pi * np.random.rand()  # Random azimuthal angle 

        # Cartesian positions
        pos[i, 0] = r * np.cos(phi)
        pos[i, 1] = r * np.sin(phi)
        pos[i, 2] = np.random.normal(0, z_spread)   # Normal pdf with scale parameter = z_spread

        # Circular orbit velocity with small dispersion
        M_enclosed = mass * (r / (r + scale_radius))
        v_circ = circular_velocity(r,M_enclosed)
        v_phi = v_circ     # Tangential velocity for circular motion

        # Convert to Cartesian velocities
        vel[i, 0] = -v_phi * np.sin(phi) + np.random.normal(scale=0.05 * v_circ)    # vx
        vel[i, 1] = v_phi * np.cos(phi) + np.random.normal(scale=0.05 * v_circ)     # vy
        vel[i, 2] = np.random.normal(0, 0.05 * v_circ)                              # vz

    return pos, vel, masses

# Script starts here
if __name__=='__main__':

    # Get N from arg
    try:
        total_particles = int(float(sys.argv[1]))
    except Exception as err:
        print(err,'\nSetting N = 1000')
        total_particles = 1000

    # Galaxy parameters
    scale_radius_kpc = 2.0 # in kpc
    scale_radius = scale_radius_kpc*3.086e19 # in m 
    galaxy_mass = 1*10**40  # Total galaxy mass in kg
    characteristic_time = np.sqrt((scale_radius**3)/(G_si*galaxy_mass)) # characteristic scale time
    print("Characteristic time: ",characteristic_time/(3600*24*365*10**6),' Myr')
    z_spread = 1.2e19 / scale_radius    # convert z_spread to units of scale_radius

    # Generate the galaxy
    pos, vel, mass = create_disk(total_particles, 1, 1, z_spread)

    # Save to .npy files
    np.save('init_pos.npy', pos)
    np.save('init_vel.npy', vel)
    np.save('init_mass.npy', mass)

    # Save galaxy and unit info
    info_str = str(total_particles) + '\n' + str(scale_radius) + '\n' + str(galaxy_mass) + '\n' + str(characteristic_time) + '\n' + str(0) + '\n' + str(0) + '\n' + str(0)
    info = open(f"init_info.txt", "w")
    info.write(info_str)
    info.close()
    print(info_str)
    ## Info_str ##
    # Num particles
    # Scale radius (m)
    # Galaxy mass (kg)
    # Characteristic time (s)
    # Separation (m) = 0    (updated after setting up collision)
    # Collision angle = 0   (updated after setting up collision)
    # Time to impact (s) = 0    (updated after setting up collision)


    ## Plotting ##

    x, y, z = pos.T
    # Convert to Kpc
    x =(x*scale_radius) /3.086e19
    y =(y*scale_radius)/3.086e19
    z =(z*scale_radius) /3.086e19

    # x, y
    pos_unit = 'kpc'
    plt.style.use('dark_background')
    fig, ax =plt.subplots()
    ax.scatter(x,y,s=1,c='white',label='disk',zorder=1,marker='.',edgecolors='none')
    ax.set_title(f'N = {total_particles}')
    ax.set_aspect('equal','box')
    ax.set_xlim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_ylim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_xlabel(f'x ({pos_unit})')
    ax.set_ylabel(f'y ({pos_unit})')
    fig.tight_layout()
    fig.savefig('ICs_xy.png',dpi=400)
    plt.close()

    # y, z
    plt.style.use('dark_background')
    fig, ax =plt.subplots()
    ax.scatter(y,z,s=1,c='white',label='disk',zorder=1,marker='.',edgecolors='none')
    ax.set_title(f'N = {total_particles}')
    ax.set_aspect('equal','box')
    ax.set_xlim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_ylim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_xlabel(f'y ({pos_unit})')
    ax.set_ylabel(f'z ({pos_unit})')
    fig.tight_layout()
    fig.savefig('ICs_yz.png',dpi=400)
    plt.close()

    # 3D plot
    plt.style.use('dark_background')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,s=2,c='white',label='disk',zorder=1,marker='.',edgecolors='none')
    fig.tight_layout()
    ax.set_xlim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_ylim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_zlim([-5*scale_radius_kpc,5*scale_radius_kpc])
    ax.set_aspect('equal','box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.rcParams['axes.facecolor']=='black'
    plt.axis('off')
    fig.savefig('ICs_3D.png',dpi=400)
    plt.close()

    # v(r)
    vel *= (scale_radius/characteristic_time)* 3600/1000      # Convert velocities to km/h
    radials = np.sqrt(x**2 + y**2)# + z**2)
    radial_v = np.sqrt(vel[:,0]**2 + vel[:,1]**2)# + vel[:,2]**2)
    index = np.argsort(radials)
    plt.style.use('default')
    fig, ax =plt.subplots()
    ax.scatter(radials,radial_v,s=1,c='blue',label='disk',zorder=0,marker='.')
    ax.set_title(f'N = {total_particles}')
    ax.set_xlim([0,10*scale_radius_kpc])
    ax.set_xlabel(f'r ({pos_unit})')
    ax.set_ylabel(f'v (km/h)')
    fig.savefig('ICs_r_v.png',dpi=400)
    plt.close()
