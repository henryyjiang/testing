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
        train_config = 'mdl_config.yml'
        self.calculator = MDLCalculator(config=train_config)
        self.best_losses = []
        self.best_loss = float('inf')
        self.avg_losses = []

    def init_atoms(self, density=0.2):
        beta = np.random.uniform(0, 180)
        gamma = np.random.uniform(0, 180)
        minCosA = - np.sin(gamma * np.pi/180) * np.sqrt(1 - np.cos(beta* np.pi/180) ** 2) + np.cos(beta * np.pi/180) * np.cos(gamma * np.pi/180)
        maxCosA = np.sin(gamma * np.pi/180) * np.sqrt(1 - np.cos(beta* np.pi/180) ** 2) + np.cos(beta * np.pi/180) * np.cos(gamma * np.pi/180)
        alpha = np.random.uniform(minCosA, maxCosA)
        alpha = np.arccos(alpha) * 180 / np.pi
        a = np.random.rand() + .000001
        b = np.random.rand() + .000001
        c = np.random.rand() + .000001
        cell=[a, b, c, alpha, beta, gamma]

        atoms = Atoms(self.composition, cell=cell, pbc=(True, True, True))
        vol = atoms.get_cell().volume

        ideal_vol = len(self.composition) / density
        scale = (ideal_vol / vol) ** (1/3)
        cell = [scale * a, scale * b, scale * c, alpha, beta, gamma]
        atoms.set_cell(cell)
        scaled_positions = np.random.uniform(0., 1., (len(atoms), 3))
        atoms.set_scaled_positions(scaled_positions)

        return atoms

    def dimensions_to_atoms(self, params):
        cell = params[:9].reshape(-1, 3)
        positions = params[9:].reshape(-1, 3)
        atoms = Atoms(self.composition, cell=cell, pbc=(True, True, True), positions=positions)
        return atoms

    def objective_func(self, params):
        atoms = self.dimensions_to_atoms(params)

        atoms.set_calculator(self.calculator)
        loss = atoms.get_potential_energy()

        if loss < self.best_loss:
            self.best_loss = loss

        return loss

    def f(self, x):
        """Higher-level method to do forward_prop in the
        whole swarm.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        n_particles = x.shape[0]
        #j = [self.objective_func(x[i]) for i in range(n_particles)]

        """
        upload as batch, use mdl_ff optimize in order to perform local optimization and return loss.
        how do i pass the resulting atoms into my pso function?
        """
        my_dataset = download_dataset(repo="MP", save=True)
        my_dataset = json.load(open("../data/data_subset_msp.json", "r"))
        train_config = 'mdl_config.yml'
        forcefield = MDL_FF(train_config, my_dataset)

        new_atoms = [self.dimensions_to_atoms(x[i]) for i in range(n_particles)]
        objective_func = Energy(normalize=True, ljr_ratio=1)
        new_atoms, obj_loss, energy_loss, novel_loss, soft_sphere_loss = forcefield.optimize(new_atoms, steps=100, objective_func=objective_func, log_per=0,
                                                                                             learning_rate=.05, batch_size=4, cell_relax=True, optim="Adam")

        #self.best_losses.append(self.best_loss)
        #self.avg_losses.append(np.mean(j))

        #return np.array(j)
        res = [i[0] for i in obj_loss]
        return res
    def run(self):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9} # cognitive, social, itertia
        particles = 10 # number of particles in system
        dimensions = 9 + len(self.composition)*3 # first 9 are cell, rest are atom positions

        init_positions = np.empty((particles, dimensions))
        for i in range(particles):
            init_atoms = self.init_atoms()
            flattened_cell = [i for l in init_atoms.get_cell().tolist() for i in l]
            flattened_pos = [float(i) for l in init_atoms.positions for i in l]
            init_pos = flattened_cell + flattened_pos
            init_positions[i] = np.array(init_pos)
            init_atoms.set_calculator(self.calculator)
            loss = init_atoms.get_potential_energy()
        filename = "init" + "_structure" + ".cif"
        ase.io.write(filename, self.dimensions_to_atoms(init_positions[0]))

        optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=dimensions, options=options, init_pos=init_positions)

        # Perform optimization
        cost, pos = optimizer.optimize(self.f, iters=1000)

        filename = "best" + "_structure" + ".cif"
        ase.io.write(filename, self.dimensions_to_atoms(pos))

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