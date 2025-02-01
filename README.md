# Parallel Direct N-Body Simulations of Galaxy Mergers 



https://github.com/user-attachments/assets/5c5bf1ed-c487-4535-874b-e2f47c743f33



This project explores direct gravitational N-body simulations in the context of simple galaxy mergers using distributed and shared memory methods. Two serial programs are included (Python, Cython), an OpenMP program (via Cython), and two MPI programs (pure Python, and OpenMP via Cython). The MPI Cython script allows for hybrid OpenMP + MPI parallelism. Detailed instructions are outlined below, but note that there are also instructions and program info at the top of every script.


#### Dependencies
This project has been built and tested on Ubuntu. A list of all dependencies can be found in the [environment.yml](environment.yml) file, and the environment can be reconstructed in conda using:
```
conda env create -f environment.yml
```

#### Compiling Cython 
Cython extension building is required in all three [SingleProcess/](SingleProcess), [OpenMP/](OpenMP), and [MPI/](MPI). These are to be compiled in the usual way, and setup scripts have been included specific to each folder. 

E.g., to create the C extension for MPI, do:
```
cd GalaxyMerger/MPI/
python setup.py build_ext -fi
```
Repeat this for [SingleProcess/](SingleProcess) and [OpenMP/](OpenMP). These will create the leapfrog.c extension for each folder.

## Initial Conditions ## 

The script [make_galaxy.py](make_galaxy.py) creates a simplified disk galaxy of N particles that must be subsequently relaxed over 20,000+ timesteps. Note that simulations units are used ($M=G=L=1$, where $M$ is the galaxy mass, $G$ is the gravitational constant, and $L$ is a characteristic scale length of the system - taken as the scale radius of the galaxy). Parameters such as scale radius, mass, etc are hard-coded and can be changed on [line 67 in make_galaxy.py](https://github.com/CorboPy/GalaxyMerger/blob/7daefc3bddd869114ff9f2b590eb10de0b169fad/make_galaxy.py#L67) 

E.g. to create a galaxy of 1000 particles, do:

```
python make_galaxy.py 1000
```

This creates .npy files for particle masses, positions, and velocities, and saves them in GalaxyMerger/. An info.txt is also created, which contains info on scale radius, scale height, etc (see [line 89 in make_galaxy.py](https://github.com/CorboPy/GalaxyMerger/blob/7daefc3bddd869114ff9f2b590eb10de0b169fad/make_galaxy.py#L89)).

To relax the galaxy, simply run any of the programs without an argument. Ensure that you are relaxing over a sufficient number of timesteps.

E.g., to relax your galaxy using the OpenMP program, do:

```
python N_body_OpenMP.py
```

If successful, the positions, velocities, and a .log file containing simulation info, will be saved to OpenMP/output.
When satisfied that the galaxy is sufficiently relaxed, [setup_collision.py](setup_collision.py) can be used to set up a collision between two of the same galaxy. Before running, adjust collision parameters accordingly on [line 150](https://github.com/CorboPy/GalaxyMerger/blob/7daefc3bddd869114ff9f2b590eb10de0b169fad/setup_collision.py#L151). Galaxy 2 may also be rotated around the x-axis by setting a non-zero ```theta_rot```.
Next, run setup_collision.py with the positions .npy file from your relaxed simulation as the argument.

E.g., if the galaxy was relaxed using OpenMP, do:

```
cd GalaxyMerger/OpenMP/
python setup_collision.py OpenMP/output/DD-MM-YYYY_HH-MM-SS_all_positions.npy
```

This will save a new set of initial merger conditions to a new folder called initial_galaxy_N, where N is the number of particles in the combined system of galaxies (i.e. 2000 if your initial galaxy was 1000).
Now, the merger simulation can be run using the instructions below.

## Serial 
To run any of the serial codes, they must be run from within [SingleProcess/](SingleProcess) and supplied with the full or relative path to one of the folders containing the initial conditions. If you wish to change simulation parameters such as the number of timesteps, the number of saved timesteps, the step size, or the value for the softening parameter, these can be found hard-coded under the main() function in each program.

E.g. to run the serial program that utilizes the C extension on N=800 initial conditions, do:

```
cd GalaxyMerger/SingleProcess/
python N_body_Single_Cython.py ../initial_galaxy_800
```

The simulation parameters should be printed to the screen before the code enters the main loop. If successful, the positions, velocities, and a .log file containing simulation info, will be saved to SingleProcess/output. To animate the results, see the subsection on [Plotting and Animating](#plotting-and-animating).

## OpenMP 
To run the OpenMP program, they must be run from within [OpenMP/](OpenMP) and supplied with the full or relative path to one of the folders containing the initial conditions. If you wish to change simulation parameters such as the number of timesteps, the number of saved timesteps, the step size, or the value for the softening parameter, these can be found hard-coded under the main() function in each program. If you wish to change the number of OpenMP threads, these can be found a few lines into the code as the variable ```num_omp_threads```, which has been set to 8 as a default.

E.g. to run the OpenMP program on N=800 initial conditions, do:

```
cd GalaxyMerger/OpenMP/
python N_body_OpenMP.py ../initial_galaxy_800
```

The simulation parameters should be printed to the screen before the code enters the main loop. If successful, the positions, velocities, and a .log file containing simulation info, will be saved to OpenMP/output. To animate the results, see the subsection on [Plotting and Animating](#plotting-and-animating).

## MPI + OpenMP 
To run the MPI programs, they must be run from within [MPI/](MPI) and supplied with the full or relative path to one of the folders containing the initial conditions. If you wish to change simulation parameters such as the number of timesteps, the number of saved timesteps, the step size, or the value for the softening parameter, these can be found hard-coded under the main() function in each program. For [N_body_MPI_Cython.py](MPI/N_body_MPI_Cython.py), if you wish to change the number of OpenMP threads, these can be found a few lines into the code as the variable ```num_omp_threads```, which has been set to 2 as a default.

E.g. to run the MPI Cython program on N=800 initial conditions over 8 processes, do:

```
cd GalaxyMerger/MPI/
mpiexec -np 8 python N_body_MPI_Cython.py ../initial_galaxy_800
```

The simulation parameters should be printed to the screen before the code enters the main loop. If successful, the positions, velocities, and a .log file containing simulation info, will be saved to MPI/output. To animate the results, see the subsection on [Plotting and Animating](#plotting-and-animating).

*Note: certain ```n_p``` are rejected if it doesn't divide nicely into the number of particles in the system. Generally, 1, 2, 4, 5, 8, 10, 16, 20, 25 are safe.*

## Plotting and Animating
Inside [SingleProcess/](SingleProcess), [OpenMP/](OpenMP), and [MPI/](MPI), there is a script called ```plot.py``` which can create and save 2D or 3D PNGs and automatically create an animation from them using FFmpeg. Additionally, it can plot potential vs kinetic energies over the whole simulation, but note that this can take a while for large N as it has to calculate the potential at each timestep.

The plot.py script works by saving PNGs of each snapshot from the simulation, before using FFmpeg to create an animation from the PNGs. Depending on how many snapshots are being saved, this can mean a lot of PNGs are being created!

*Note: multiprocessing is used to substantially speed up the plotting process.*

To run plot.py, it must be ran from the folder that the original simulation was run in. The first argument to plot.py is the full or relative path to the ```output/DD-MM-YYYY_HH-MM-SS_all_positions.npy``` file. You then must give the script one or more of the following flags:

```-2d``` for 2D Plot + animation, 

```-3d``` for 3D Plot + animation, 

```-e``` for energies plot, 

```-s``` for a combined plot of 4 snapshots, 

```-a``` to do all of the above.

E.g., to create a 2D animation, 3D animation, combined snapshot pdf plot, and energies plot for an MPI simulation, do:

```
cd GalaxyMerger/MPI/
python plot.py output/DD-MM-YYYY_HH-MM-SS_all_positions.npy -a
```

As the program is running, processes are assigned to speed up the plotting process (this should not exceed the number of physical cores on your system). If successful, you should see a new folder ```DD-MM-YYYY_HH-MM-SS/``` inside MPI/plot/. Here, you can view the animations, entitled ```movie.mp4``` and ```movie_3d.mp4```, the snapshot plot, entitled ```row_snapshots.pdf```, and the ```energies.png``` plot. Additionally, the potential energies at each step are saved to a .npy file to prevent the need for recalculating every time you wish to edit the plot formatting.

