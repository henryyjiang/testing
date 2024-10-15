from ase import Atoms
import ase
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from matdeeplearn.common.ase_utils import MDLCalculator
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("../.."))))
from msp.utils.objectives import Energy
from msp.forcefield import MDL_FF
import json
from msp.dataset import download_dataset



class PSO():
    def __init__(self):
        self.composition = [22, 22, 8, 8, 8, 8]
        self.cell = [5.56641437, 5.56641437, 5.56641437, 140.05412510, 140.05412510, 57.77112661]
        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []
        self.best_pos = Atoms(self.composition, cell=self.cell, pbc=(True, True, True))

        my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
        train_config = 'mdl_config.yml'
        self.forcefield = MDL_FF(train_config, my_dataset)
        self.energy = Energy(normalize=True, ljr_ratio=1)

        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} # cognitive, social, inertia
        particles = 10 # number of particles in system
        dimensions = len(self.composition)*3 # first 9 are cell, rest are atom positions

        init_positions = np.empty((particles, dimensions))
        for i in range(particles):
            init_atoms = self.init_atoms()
            init_pos = [float(i) for l in init_atoms.positions for i in l]
            init_positions[i] = np.array(init_pos)
        filename = "init" + "_structure" + ".cif"
        ase.io.write(filename, self.dimensions_to_atoms(init_positions[0]))

        self.optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions)

    def init_atoms(self, density=0.2):
        atoms = Atoms(self.composition, cell=self.cell, pbc=(True, True, True))

        scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        atoms.set_scaled_positions(scaled_positions)

        return atoms

    def dimensions_to_atoms(self, params):
        positions = params.reshape(-1, 3)
        atoms = Atoms(self.composition, cell=self.cell, pbc=(True, True, True), positions=positions)
        return atoms

    def atoms_to_dimensions(self, atoms):
        pos = [float(i) for l in atoms.positions for i in l]
        return np.array(pos)

    def obj_func(self):
        positions = self.optimizer.swarm.position
        new_atoms = [self.dimensions_to_atoms(positions[i]) for i in range(len(positions))]
        new_atoms, obj_loss, energy_loss, novel_loss, soft_sphere_loss = self.forcefield.optimize(new_atoms, steps=100,
                                                                                             objective_func=self.energy,
                                                                                             log_per=0,
                                                                                             learning_rate=.05,
                                                                                             batch_size=4,
                                                                                             cell_relax=True,
                                                                                             optim="Adam")

        self.optimizer.swarm.position = np.array([self.atoms_to_dimensions(new_atoms[i]) for i in range(len(new_atoms))])

        return [i[0] for i in energy_loss]

    def f(self, x):
        j = self.obj_func()

        for i in range(len(j)):
            if j[i] < self.best_loss:
                self.best_pos = self.dimensions_to_atoms(x[i])
                self.best_loss = j[i]
        self.best_losses.append(self.best_loss)
        self.avg_losses.append(np.mean(j))

        return np.array(j)

    def run(self):
        cost, pos = self.optimizer.optimize(self.f, iters=500)

        filename = "best" + "_structure" + ".cif"
        ase.io.write(filename, self.best_pos)

        plt.plot(self.best_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Best Loss')
        plt.title('Best Losses')
        plt.show()
        plt.close()

        plt.plot(self.avg_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Average Loss')
        plt.title('Average Losses')
        plt.show()
        plt.close()

if __name__ == "__main__":
    pso = PSO()
    pso.run()