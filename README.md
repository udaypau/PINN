# Physics-Integrated Neural Networks (PINNs) for Solving Navier-Stokes Flow Problems

Welcome to the official repository for my final-year engineering project! This repository explores the intersection of deep learning and fluid dynamics by utilizing Physics-Informed Neural Networks (PINNs) to solve forward and inverse Navier-Stokes problems. It also includes a comprehensive collection of all the reference materials, datasets, and resources I have used throughout my research so far.

## The Vision
Traditional Computational Fluid Dynamics (CFD) solvers are incredibly powerful, but they can be computationally expensive and often struggle when dealing with sparse, noisy, or incomplete data. This project takes an alternative approach: rather than relying solely on massive training datasets, we embed the fundamental physical laws of fluid mechanics—like the conservation of mass and momentum—directly into the neural network's loss function. By forcing the AI to strictly respect the governing physics, the network can reconstruct complex, non-linear flow fields even when the provided data is murky or limited.

## What's Inside
Built entirely in PyTorch, this repository serves as a modular playground and testing framework for PINN mechanics. It currently features:
* **Canonical Flow Solvers:** Full training pipelines for 2D Hagen-Poiseuille pipe flow and unsteady lid-driven cavity flow ($Re=100$).
* **Rigorous CFD Validation:** Baseline datasets and point-cloud comparisons generated via OpenFOAM to verify the network's accuracy against established numerical methods.
* **Advanced PINN Mechanics:** Implementations of dynamic weight annealing to balance multi-objective loss functions, as well as time-windowing strategies to address the breakdown of temporal causality in irreversible viscous flows.
* **Decoupled Architecture:** Clean separation between YAML-based training configurations and lightweight, standalone inference scripts.
* **Project Resources:** All academic papers, reference codes, OpenFOAM configuration files, and raw CSV datasets utilized during the project's development.

## 🤝 Open Source & Contributions
This project is entirely open-source, built with the hope that it can serve as a helpful baseline or reference for other students, researchers, and engineers curious about scientific machine learning. Please feel absolutely free to fork this repository, use the code for your own experiments, adapt the architectures, or build something entirely new on top of it.

Because this is a student project and a continuous learning journey, help, advice, and contributions are deeply appreciated! Whether you want to point out a bug, suggest a code optimization, improve the documentation, or larp on the mathematical nuances of phase space contraction in 6d dimension ultra quasi continuum of competitive performative feminism , I would love to hear from you.

Feel free to open an issue, submit a pull request, or just reach out. Let's learn and build together!

