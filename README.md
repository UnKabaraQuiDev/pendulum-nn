# Pendulum Neural Network Project

---

![img](https://raw.githubusercontent.com/UnKabaraQuiDev/pendulum-nn/refs/heads/main/imgs/breakthrough.png)

---

# How to run
## Requirements

- Java 17+
- Maven

### Option 1: Using Maven

```bash
mvn clean compile exec:java
````

### Option 2: Build and run manually

```bash
mvn clean package
java -jar target/pendulum-nn.jar
```

## Configuration

On the first run, it will create a config inside `config/`, edit it or leave it as-is and restart the program.

## LWJGL Natives

LWJGL requires native libraries depending on your OS.

Supported systems:

* Linux
* (probably works on windows too but hasn't been tested)

---

# Documentation
## 1. Project Type
This project is a custom GPU-based simulation and learning system.  
It combines real-time physics simulation, neural networks, and a genetic algorithm.  
All learning is evolutionary; no gradient-based training is used.

The system is implemented in Java and GLSL compute shaders and runs fully on the GPU, with minimal CPU intervention.

---

## 2. Project Idea / Goal
The goal of this project is to train multiple neural-network agents to balance a pendulum mounted on a moving cart.

Each agent:
- Observes the physical state of its cart and pendulum
- Outputs a cart acceleration
- Is evaluated based on stability and position over time

Agents evolve over generations using selection, crossover, and mutation, improving their behavior without supervision.

---

## 3. Technology & Stack

### Languages
- Java
- GLSL (Compute Shaders)

### Libraries & Engine
- Custom Standalone Game Engine (`lu.kbra.standalone.gameengine`)
- OpenGL (Compute Shaders, SSBOs)
- JOML (math library for vectors and matrices)
- Jackson (JSON serialization)

### Execution Model
- GPU-first architecture
- Parallel simulation of up to >10,000 agents
- Fixed-timestep physics

---

## 4. System Overview

The system is composed of three main stages per simulation step:

1. **Neural Network Forward Pass (GPU)**
2. **Physics Simulation + Fitness Evaluation (GPU)**
3. **Evolutionary Selection and Reproduction (CPU)**

![img](https://raw.githubusercontent.com/UnKabaraQuiDev/pendulum-nn/refs/heads/main/imgs/schema.svg)

---

## 5. Neural Network Design

Each agent owns a small fully connected neural network.

### Structure
- Inputs: 5
- Hidden layers: 3 layers of 1 neuron each
- Outputs: 1
- Activation function: `tanh`

### Inputs
1. Cart position (`x`)
2. Cart velocity (`vx`)
3. Pendulum angle
4. Pendulum angular velocity
5. Normalized cart position relative to bounds

### Output
- Cart acceleration (scaled and clamped)

Weights and biases are stored in GPU buffers and processed entirely in compute shaders.

---

## 6. Physics Simulation

Physics is computed in a compute shader for each agent independently.

### Simulated Components
- Cart movement with friction and boundary limits
- Pendulum rotation under gravity
- Angular friction
- Acceleration clamping

### Timing
- Fixed update rate: 60 UPS
- Episode duration: 300 virtual seconds
- Total steps per generation: 18,000

---

## 7. Fitness (Grading) System

Each agent accumulates a fitness score over time.

Fitness rewards:
- Upright pendulum angle
- Staying near the center of the track
- Smooth, stable motion

Fitness is integrated over time:`grade += score * deltaTime`


This favors agents that remain stable for long periods.

---

## 8. GPU Buffer Architecture

Each agent is represented by slices of large GPU buffers:

- **Input buffer**: neural network inputs
- **Output buffer**: neural network outputs
- **Physics buffer**: position, velocity, angle, angular velocity
- **Weights buffer**: NN weights
- **Biases buffer**: NN biases
- **Grade buffer**: accumulated fitness
- **Transform buffer**: model matrices for instanced rendering

All agents share the same buffers, indexed by agent ID.

---

## 9. Compute Shaders

### Neural Network Compute Shader
Responsibilities:
- Read inputs, weights, and biases
- Perform full forward pass
- Write output acceleration

Each shader invocation processes exactly one agent.

---

### Postprocess / Physics Compute Shader
Responsibilities:
- Apply acceleration to cart
- Update physics state
- Compute fitness contribution
- Generate next NN inputs
- Write model matrix for rendering

This shader fully replaces CPU-side physics and state updates.

---

## 10. Evolutionary Algorithm

At the end of each generation:

1. Fitness scores are read back from the GPU
2. Top-performing agents are selected
3. Survivors are saved to disk (JSON)
4. New population is created using:
   - Crossover (gene-wise selection)
   - Mutation (Gaussian noise)
5. New agents are uploaded to the GPU
6. Simulation restarts with new initial conditions

Only the evolutionary logic runs on the CPU.

---

## 11. Initialization & Reset Logic

At startup or between generations:
- Physics states are randomized using Gaussian noise
- Grades, inputs, and outputs are cleared
- A single postprocess step initializes valid NN inputs

Previous generation data can be reloaded to continue training.

---

## 12. Rendering & Visualization

Rendering is optional and decoupled from simulation speed.

Features:
- Instanced rendering for carts and pendulums
- Optional display of top agents only
- Vertical separation of agents for visual clarity

Rendering does not affect learning or physics.

---

## 13. Performance Characteristics

- Thousands of agents simulated in parallel
- No per-agent CPU cost during simulation
- Deterministic fixed-timestep physics
- Compute shader local size chosen dynamically based on GPU limits

---

## 14. Conclusion

This project demonstrates a fully GPU-driven evolutionary learning system.  
By combining compute shaders, physics simulation, and genetic algorithms, it achieves large-scale parallel learning without traditional machine learning frameworks.

The architecture is flexible and can be extended to other control problems beyond the pendulum system.

---

## 15. Lessons Learned

- GPU memory layout is critical for performance
- Evolutionary methods scale well on massively parallel hardware
- Debugging compute shaders requires careful instrumentation
- Separating simulation and rendering simplifies system design

---

## 16. Possible Next Steps

- Increase NN size or depth
- Add curriculum learning with changing environments
- Introduce multi-objective fitness
- Add inter-agent competition or cooperation
- Export trained agents for standalone inference

---

## 17. Dependencies & Execution

- OpenGL 4.3+ (compute shader support)
- Compatible GPU with sufficient SSBO limits
- Java runtime with OpenGL bindings

No external ML frameworks are required.
