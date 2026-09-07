# Elastica-RL: RL Control of Soft Pneumatic Manipulators

PPO policies for two underactuated soft pneumatic robots, simulated as Cosserat rods in
[PyElastica](https://github.com/GazzolaLab/PyElastica): 
- a three-bellow continuum trunk reaching to
targets across its workspace, 
- and a dual-finger gripper closing onto a target.

<p align="center">
  <img src="demo/trunk.gif" width="45%"/>
  <img src="demo/gripper.gif" width="45%"/>
</p>

## Why a 1D simulator

Volumetric FEM (ANSYS, Abaqus) is accurate but far too slow to generate the millions of transitions
PPO needs, and is not differentiable. Cosserat rod theory collapses the 3D actuator to a 1D
centerline, bringing a 0.1 s control step down to milliseconds. 2M timesteps becomes an overnight
CPU job across 8 workers.

## Pressure to curvature

PyElastica has no fluidic chambers, so pressure is mapped to intrinsic rod curvature through a
trigonometric Jacobian from the PCC framework. The agent commands three chamber pressures rather
than the tip, which keeps the real robot's underactuation intact:

```
kappa_x = k_max * ( P1 - 0.5*P2 - 0.5*P3 )
kappa_y = k_max * ( sqrt(3)/2 * P2 - sqrt(3)/2 * P3 )
```

`k_max` is calibrated against static pressure-deformation curves from a volumetric ANSYS model, so
tip deflection at 100 kPa matches between the two simulators.

Actions are clipped to non-negative pressure. In a `[-1, 1]` space a freshly initialized network
outputs ~0, which maps to half-inflating all three chambers at once. That is co-contraction: on
hardware it extends and stiffens the arm, and in a pure-bending 1D rod it collapsed the policy into
a locked state.

## Randomization

Three layers, resampled every episode:

- **Goal space**: target yaw over `[0, 2pi]`, curvature over `[0.2, 2.5] m^-1`, sampled across the
  true kinematic workspace. The agent has to learn a continuous inverse kinematics mapping instead
  of memorizing trajectories, which is what prevents directional mode collapse.
- **Dynamics**: Young's modulus +/-20%, damping +/-50%, covering silicone batch variation and
  material degradation.
- **Domain**: pneumatic latency as a low-pass filter on commanded pressure,
  `P = (1-a)*P + a*P_target`, `a` sampled from `[0.4, 1.0]`. Real air lines lag; standard Cosserat
  actuation assumes bending moments apply instantly. Current pressure is part of the observation,
  which keeps the problem Markovian under the filter.

## What randomization changes

Without it, the policy converges in 500k timesteps and looks excellent. It has actually learned to
whip: hold off actuating until the last moment, then strike the target. That only works because
`a = 1.0` makes inflation instant. Add any real latency at evaluation and it overshoots or stalls.

With randomization the trunk needs roughly 2M timesteps for the critic to stabilize against the
stochastic physics, and converges to conservative braking that holds whether the episode draws a
stiff rod with sluggish air or a compliant rod with instant air. Mean episodic reward is ~911,
against a ~1600 bound that assumes teleporting to the target on frame zero.

## Rewards

- Trunk: potential shaping on change in tip-target distance, an absolute distance penalty, a
distance-scaled bonus inside a 5 cm zone, and a tip-velocity penalty for braking.

- Gripper: summed fingertip distances, precision bonus, a symmetry penalty on `|x1 + x2|` to keep the
grasp centered, velocity damping, and a hard cross-over penalty so the two rods stay out of
non-physical overlap without a contact model.

## Layout

```
envs/pneumatic_arm_env.py    3-bellow trunk, 12D obs, 3D action
envs/soft_gripper_env.py     dual-finger gripper, 15D obs, 2D action
encoder.py                   shared MLP feature extractor
train_elephant_trunk.py      PPO, 8 SubprocVecEnv workers, VecNormalize
train_gripper.py
evaluate_*.py                deterministic rollout + 3D animation
```

Trunk observation is tip position, target error vector, tip velocity, and chamber pressures.
Episodes run 100 control steps (10 simulated seconds) at dt = 1e-4.

## Running

```bash
uv sync
uv run train_elephant_trunk.py
uv run evaluate_elephant_trunk.py runs/trunk/run_<timestamp> --save demo/trunk.gif
tensorboard --logdir runs/
```

`VecNormalize` statistics save next to each checkpoint and are needed at evaluation time.