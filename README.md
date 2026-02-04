# PALogic (lu.kbra.pendulum_nn) — How it works

## What PALogic is
`PALogic` is a `GameLogic` implementation that trains many neural-network “agents” in parallel on the GPU.

Each agent controls a cart (position `x`, velocity `vx`) with a pendulum (angle `a`, angular velocity `va`).  
The NN outputs a single value per agent: **cart acceleration**. A compute shader simulates physics, updates inputs, and accumulates a **fitness score** (“grade”). After a fixed virtual time, the code keeps the best agents, creates a new generation by crossover + mutation, and repeats.

Rendering is mostly for visualization: each agent is drawn as an instanced mesh, and “top agents” are highlighted by limiting the triangle instance count.

---

## Key constants and what they mean

### Simulation timing
- `UPS = 60`
- `FIXED_D_TIME = 1 / UPS`
- `VIRTUAL_SECONDS = 300`
- `MAX_ITERATIONS = UPS * VIRTUAL_SECONDS`  
So one generation runs for `300s * 60 = 18,000` iterations on a fixed timestep.

### Population / batching
- `TOP_AGENTS = 20` (survivors used as parents)
- `AGENT_BATCHES = 1`
- `MAX_AGENTS = 10_000`
- `instanceCount = min(MAX_AGENTS, AGENT_BATCHES * AGENT_PER_BATCHES)`
- `AGENT_PER_BATCHES = LOCAL_SIZE.x * LOCAL_SIZE.y * LOCAL_SIZE.z`  
`LOCAL_SIZE` is chosen at runtime from GL limits (`computeOptimalComputeShaderLocalSize()`).

### Evolution
- `MUTATE_RATE = 0.5`
- `MUTATE_STRENGTH = 1`

### Physics / control
- `GRAVITY = 9`
- `MASS = 10`
- `PENDULUM_LENGTH = 1`
- `WIDTH = 3` (track width; bounds are `[-WIDTH/2, +WIDTH/2]`)
- `FRICTION = 0.25` (cart)
- `ANGULAR_FRICTION = 0.4` (pendulum)
- `ACCELERATION_FACTOR = 10`
- `ACCELERATION_MAX = 5` (acceleration clamp)

---

## NN structure (CPU-side)
`struct` is an `NNStructure`:
- `inputCount = 5`
- `innerLayers = {1, 1, 1}`
- `outputCount = 1`
- activation: `TANH`

`NNStructure` provides:
- `computeWeightCount()` = sum of fully-connected weights across layers
- `computeBiasCount()` = sum of biases across layers
- `computeNeuronCount()` = inputs + all hidden + outputs

`NNInstance` is just:
- `NNStructure structure`
- `float[] weights`
- `float[] biases`

Instances are serialized as JSON (`top.XXXXX.json`) using Jackson.

---

## GPU buffers (attrib arrays)
PALogic stores all agent data in GPU buffers.

### Buffer indices (important bindings)
- `INPUT_IDX = 0`
- `OUTPUT_IDX = 1`
- `PHYSICS_IDX = 2`
- `WEIGHTS_IDX = 3`
- `BIASES_IDX = 4`
- `TRANSFORMS_IDX = InstanceEmitter.TRANSFORM_BUFFER_INDEX`
- `GRADE_IDX = InstanceEmitter.FIRST_BUFFER_INDEX`

### Actual buffers
- `weightsValueArray : SyntheticFloatAttribArray`
  - length = `struct.computeWeightCount() * instanceCount`
- `biasesValueArray : SyntheticFloatAttribArray`
  - length = `struct.computeBiasCount() * instanceCount`
- `inputNeuronsValueArray : SyntheticFloatAttribArray`
  - length = `struct.inputCount * instanceCount` (5 per agent)
- `outputNeuronsValueArray : SyntheticFloatAttribArray`
  - length = `struct.outputCount * instanceCount` (1 per agent)
- `physicsVec4sValueArray : SyntheticVec4fAttribArray`
  - length = `2 * instanceCount`
  - layout per agent:
    - `states[agent*2 + 0] = vec4(x, angle, 0, 0)`
    - `states[agent*2 + 1] = vec4(vx, angleVel, 0, 0)`
- `transformsValueArray : SyntheticMat4fAttribArray`
  - length = `instanceCount`
  - per agent `mat4 model[]` used for instanced drawing
- `gradeNeuronsValueArray : SyntheticFloatAttribArray`
  - length = `instanceCount`
  - accumulates fitness over the episode

`Synthetic*AttribArray` classes exist to create buffers of a chosen length without needing a backing CPU array. They mostly set `length` and expose `gen/init/update/read`.

---

## The two main compute shaders

### 1) NN forward pass — `nn_compute.comp`
Bound SSBOs:
- binding 0: `nnInput[]` (float)
- binding 1: `weights[]` (float)
- binding 2: `biases[]` (float)
- binding 3: `nnOutput[]` (float)

Uniforms set by Java:
- `inputSize` (5)
- `layerCount` (`innerLayers.length + 1`, because the output layer also counts as inner layer)
- `layerSize[]` (inner layers + output layer)
- `weightOffsetPerInstance`
- `biasOffsetPerInstance`
- `instanceCount`
- `activationFunction`

How it runs:
- Each global invocation handles one agent (defined by `gl_GlobalInvocationID`).
- It loads the 5 inputs for that agent.
- It performs a full forward pass across all layers.
- It writes the final activations to `nnOutput` (1 float per agent, the acceleration of the cart).

Notes:
- Checks `isnan/isinf` on sums and resets to 0.

### 2) Physics + grading + next inputs — `nn_postprocess.comp`
Bound SSBOs:
- binding 3: `nnOutput[]` (read)
- binding 4: `states[]` (read/write)
- binding 5: `model[]` (write)
- binding 0: `nnInput[]` (write)  ← prepares next NN inputs
- binding 9: `nnGrade[]` (read/write) ← fitness accumulation

Uniforms set by Java:
- `dTime`
- `instanceCount`
- `gravity`, `pendulumLength`, `mass`
- `bounds` (track min/max)
- `friction`, `angularFriction`
- `accBounds` ([-ACCELERATION_MAX, +ACCELERATION_MAX])
- `accFactor`
- user/external force fields are set but currently passed as zeros in Java
- `debugPerfectScore` (optional debugging, forces the first agent to have a perfect score by placing it in the middle facing upright at all times)

What it does per agent:
1. Read `position` and `velocity` from `states[]`.
2. Convert NN output to cart acceleration:
   - `acc = clamp(nnOutput[agent] * accFactor, accBounds.x, accBounds.y)`
3. Update cart:
   - applies friction
   - blocks movement if hitting bounds
4. Update pendulum:
   - wraps angle into `[-PI, PI]`
   - computes angular acceleration `alpha`
   - applies angular friction
5. Write transform matrix into `model[agent]`:
   - includes translation by `x`
   - also encodes a y-offset `float(agent+1)/instanceCount` (vertical separation for viewing)
6. Update grade (fitness score):
   - `nnGrade[agent] += grade() * dTime`
7. Write next-step inputs into `nnInput[]`:
   Input layout (5 floats):
   - `x`
   - `vx`
   - `angle`
   - `angleVel`
   - `relativeX` mapped to `[-1..1]`

The grade function favors “good” states using a hand-tuned expression based on `x` and `angle`.

---

## Control flow in PALogic

### init()
1. Pick compute `LOCAL_SIZE` from GL limits (`computeOptimalComputeShaderLocalSize()`).
2. Create:
   - NN compute shader
   - postprocess shader
   - clear/fill shaders for buffers
3. Create and allocate all buffers sized for `instanceCount`.
4. Clear:
   - weights, biases, grade
5. Build initial population:
   - If `RELOAD_LATEST`:
     - load latest `./output/<startTime>/top.XXXXX.json`
     - fill remaining agents by crossover+mutation of the loaded set
   - Else:
     - randomize all weights/biases
6. `upload(instances)` writes packed weights/biases to GPU buffers.
7. `resetNNs()` sets starting physics state + clears grade/inputs/outputs and runs `postProcess(FIXED_D_TIME)` once to populate the initial input buffer from physics.
8. Setup meshes + instance emitters for drawing.
9. If `REAL_TIME == false`, start “self recurring tasks” to run compute steps without tying it to frame rendering.

### resetNNs()
- Chooses new starting conditions using gaussian noise (and keeps a drifting `prevA/prevB`).
- Fills `physicsVec4sValueArray` with two default `vec4`s (position and velocity).
- Clears:
  - transforms
  - grade
  - inputs
  - outputs
- Runs `postProcess(FIXED_D_TIME)` so inputs match the starting physics.

### The per-tick loop (one simulation step)
There are two modes:

#### Non-real-time (`REAL_TIME = false`)
- `dispatchSelfRecurringTask()` posts tasks to `RENDER_DISPATCHER`:
  - `compute()` then `postProcess(FIXED_D_TIME)`
  - repeat until `iterationCount == MAX_ITERATIONS`
  - then `readBack()` and start the next generation

This is basically “run the episode as fast as possible” on the render thread dispatcher.

#### Real-time (`REAL_TIME = true`)
- Inside `render(dTime)`:
  - `compute()`
  - `postProcess(dTime)`
  - count iterations and call `readBack()` when done

### compute()
- Binds SSBOs for input/weights/biases/output.
- Uploads uniforms describing the network layout.
- Dispatches compute with enough global groups for `instanceCount`.
- Memory barrier for SSBO / buffer updates.

### postProcess(dTime)
- Binds SSBOs for output + physics + transforms + input + grade.
- Uploads physics/control uniforms.
- Dispatches compute.
- Memory barrier includes vertex attrib array barrier because `model[]` is used for instanced rendering.

---

## End of episode: selection + reproduction (`readBack()`)
When `MAX_ITERATIONS` is reached:
1. Stop scheduling (`done = true`, dispatcher cleared).
2. Read back and print stats for:
   - outputs
   - physics states
   - grades
3. Update the Swing UI (`NNFrame`) with history (avg/min/max/stddev).
4. Select survivors:
   - `topIndices = getMaxIndices(grades, TOP_AGENTS)`
   - pull each survivor’s weights and biases back from GPU
   - create `NNInstance` objects
   - `distinct()` to drop duplicate genomes
5. Save survivors asynchronously:
   - `./output/<jvmStartTime>/top.<generation>.json`
6. Update visualization:
   - `triangleEmitter.setParticleCount(uniqueTopAgents.size())`
7. Rebuild full population:
   - start with survivors
   - fill the rest by:
     - choose parents from survivors
     - `crossover(p1, p2)` (per-gene random pick)
     - `mutate(child)` (Gaussian perturbations with clamp for weights)
8. Randomize `STARTING_ANGLE` (note: it’s set but the shader-start state currently comes from `resetNNs()`’s gaussian init, not this value).
9. Upload new population to GPU.
10. Reset physics + grade + inputs and run again.

---

## Rendering (visual layer)
- Uses instanced meshes:
  - pendulum mesh (`pendulum.obj`)
  - triangle mesh (`triangle.obj`)
- Both read the same `transformsValueArray` and are rendered using two `InstanceEmitter`s.

---

## Dependencies (direct)
### Local project classes
- `NNStructure`: counts sizes for weights/biases and describes layers.
- `NNInstance`: holds weights/biases and structure for JSON persistence.
- `NNFrame`: Swing UI for grade hist + stats.
- `LimitedInstanceEmitter`: decouples draw count from allocated instance count.
- `Synthetic*AttribArray`: buffers with explicit length, plus `read/update`.

### Shaders
- `NNComputeComputeShader` → `shaders/nn_compute.comp`
- `NNPostprocessComputeShader` → `shaders/nn_postprocess.comp`
- `Clear*ComputeShader` → `shaders/clear_*.comp`
- `FillVec4fComputeShader` → `shaders/fill_vec4f.comp`

### External libs / engine
- `lu.kbra.standalone.gameengine.*` for rendering, buffers, GL wrapper, scene/camera.
- JOML (`Vector*`, `Matrix*`) for math types.
- Jackson for reading/writing `List<NNInstance>`.

---

## Things that are easy to misunderstand
- **The NN does not learn by gradients.** It’s a genetic algorithm over weights/biases.
- **Inputs are written by the postprocess shader**, not by Java. Java only seeds the physics state.
- **Fitness is accumulated over time** (`nnGrade += grade()*dTime`), so longer stability can dominate.

---

## Questions (only where the code leaves room for interpretation)
1. In `nn_postprocess.comp`, there are uniforms for external force and overwrite-cart-acceleration. Do you intend to expose those through input later (keyboard/mouse), or are they legacy?
2. The y-position in the model matrix includes `float(agent+1)/instanceCount`. Is the goal to stack agents vertically for debugging, or is it a placeholder for something else?
3. `STARTING_ANGLE` gets randomized in `readBack()`, but the reset uses gaussian drift via `prevA/prevB`. Should `STARTING_ANGLE` drive the initial state instead?
