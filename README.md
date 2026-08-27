Iron Policy v7

Multi-Agent Reinforcement Learning Tank Co-Evolution Simulator

Iron Policy v7 is a two-dimensional reinforcement learning simulation in which two AI-controlled tanks, Leo and T-90, learn while competing against each other in the same environment.

The project includes two independent PPO policies, synchronized co-evolution, mirrored evaluation, checkpoint cross-play, behavioral analysis, and an experimental Failure Memory mechanism.

Iron Policy is an experimental AI simulation. It is not intended to model or evaluate real-world weapon systems.

Key Features
Two independent PPO agents
Synchronized dual-policy co-evolution
Different physical characteristics for Leo and T-90
Mirrored training and evaluation scenarios
Random and scripted baseline agents
Checkpoint cross-play evaluation
Behavioral analysis and automatic metrics
Experimental Failure Memory mechanism
Multi-seed experiment support
Automated testing with GitHub Actions
Running on Windows
Download the repository using Code → Download ZIP.
Extract the ZIP into a normal folder.
Run iron polcy v7.vbs.

The required packages are installed automatically during the first launch.

Python 3.10–3.13 must be installed.

Alternatively, BASLAT.bat can be used.

Running on macOS and Linux

Python 3.10–3.13 is required.

Run:

sh KUR.sh

sh BASLAT.sh

KUR.sh is only required during the initial setup or when dependencies change.

Training Presets

Iron Policy provides several predefined training phases.

Smoke Test: 16,384 timesteps — 1 seed
Behavior Training: 200,000 timesteps — 1 seed
Pilot Training: 1,000,000 timesteps per seed — 3 seeds
Full Training: 5,000,000 timesteps per seed — 5 seeds

Training parameters remain consistent across systems.

Headless training runs on the CPU and does not require CUDA or an NVIDIA GPU.

Checkpoint Cross-Play

Saved checkpoints are automatically evaluated against each other after training.

A full training run can compare:

0M
1M
2M
3M
4M
final

Checkpoint cross-play can help reveal:

Performance regression against older policies
Non-transitive policy relationships
Behavioral cycles
Possible catastrophic forgetting

Checkpoint cross-play is an evaluation method only.

The current training architecture uses synchronized co-evolution and should not be described as historical or league-based self-play.

Failure Memory

Failure Memory is an experimental exploration mechanism.

When an agent enters a state similar to previously recorded failure states, the local entropy multiplier can be increased to encourage additional exploration.

Failure Memory does not directly modify the reward.

The mechanism is disabled by default.

Its effectiveness should only be evaluated through controlled experiments comparing Failure Memory OFF and ON under the same seeds and training conditions.

Experimental Scope and Limitations

The observation space contains 23 features.

Each agent observes information about the nearest hostile projectile, while additional simultaneous projectiles are not directly represented. The environment can therefore be considered partially observable from the agent's perspective.

The main reward profile is minimal. A separate shaped profile is retained for comparison and ablation experiments.

Projectile evasion measurements in new trajectory records use separate projectile distances for Leo and T-90. Older results based on a shared projectile-distance value should not be used for scientific comparison.

Testing

Iron Policy includes automated tests for the environment, training system, evaluation pipeline, Failure Memory mechanism, and other core components.

Tests are built with pytest.

GitHub Actions automatically runs the test suite on Windows and Ubuntu using Python 3.12 and 3.13 when changes are pushed to the repository.

Development Note

Iron Policy was developed extensively with the assistance of AI coding tools.

The project direction, behavioral observations, experiment decisions, system-level design changes, and iterative development goals were directed by the project author.

AI assistance was primarily used for implementation, debugging, refactoring, and translating design decisions into code.

Project Status

Iron Policy v7 is currently focused on validating its training architecture and experimental mechanisms.

The main priority is controlled experimentation rather than adding additional features.

Current evaluation priorities include:

Multi-seed experiments
Failure Memory OFF vs ON comparisons
Checkpoint cross-play
Behavioral metrics
Confidence intervals across seeds
Equal-stat control experiments
Reload-swap control experiments

Experimental claims should be based on measured results rather than assumed from the architecture alone.
