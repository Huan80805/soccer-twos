# Soccer-Twos Starter Kit

Example training/testing scripts for the Soccer-Twos environment. This starter code is modified from the example code provided in https://github.com/bryanoliveira/soccer-twos-starter.

Environment-level specification code can be found at https://github.com/bryanoliveira/soccer-twos-env, which may also be useful to reference.

## Requirements

- Python 3.8
- See [requirements.txt](requirements.txt)

## Usage

### 1. Fork this repository

git clone https://github.com/your-github-user/soccer-twos-starter.git

cd soccer-twos-starter/

### 2.1 For MAC
`bash bash env_setup.sh`

### 2.2 For Linux
```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
```

### 3. Fix protobuf and pydantic compatibility
pip install protobuf==3.20.3

pip install pydantic==1.10.13

### 4. Run `python example_random.py` to watch a random agent play the game
python example_random_players.py

### 5. Train using any of the example scripts
python example_ray_ppo_sp_still.py

python example_ray_team_vs_random.py

etc.

## Ray Notes

Switch to `init_ray()` from [utils.py](utils.py) instead of hardcoding `ray.init(...)` in each example script.

- Default behavior is plain `ray.init()`.
- To connect to a Ray cluster started from the CLI, use `SOCCER_TWOS_RAY_INIT=auto`.
- To disable the dashboard from the repo side, use `SOCCER_TWOS_RAY_INIT=no-dashboard`.

### Local Ray Package Patch

On the PACE machine used during debugging, Ray 1.4 repeatedly logged:

```text
socket.gaierror: [Errno -2] Name or service not known
```

Training still worked, but the dashboard/metrics agent was noisy because the node hostname resolution path was failing in that environment.

To suppress that warning in the `soccertwos` Conda environment, the local Ray package was patched here:

`~/miniconda3/envs/soccertwos/lib/python3.8/site-packages/ray/_private/metrics_agent.py`

The change forces Ray's Prometheus exporter to bind to `127.0.0.1`:

```python
prometheus_exporter.Options(
    namespace="ray", port=metrics_export_port, address="127.0.0.1"
)
```

Notes: The main tradeoff is that the metrics exporter is bound to localhost, so remote access to those metrics from another machine would not work without reverting the package patch.

## Agent Packaging

To receive full credit on the assignment and ensure the teaching staff can properly compile your code, you must follow these instructions:

- Implement a class that inherits from `soccer_twos.AgentInterface` and implements an `act` method. Examples are located under the `example_player_agent/` or `example_team_agent/` directories.
- Fill in your agent's information in the `README.md` file (agent name, authors & emails, and description)
- Compress each agent's module folder as `.zip`.

*Submission Policy*: Students must submit multiple trained agents to meet all assignment requirements. In both the agent desription and the report, clearly identify which agent file corresponds to each evaluation criterion (e.g., Agent1 – policy performance, Agent2 – reward modification, Agent3 – imitation learning, etc.). 

Training plots are required for every agent that is discussed or submitted. Additionally, include a direct performance comparison across agents, such as overlaid learning curves, to support your analysis.


## Testing/Evaluating

Use the environment's rollout tool to test the example agent module:

`python -m soccer_twos.watch -m example_player_agent`

Similarly, you can test your own agent by replacing `example_player_agent` with the name of your agent directory.

The baseline agent is located here: [pre-trained baseline (download)](https://drive.google.com/file/d/1WEjr48D7QG9uVy1tf4GJAZTpimHtINzE/view?usp=sharing).
To examine the baseline agent, you must extract the `ceia_baseline_agent` folder to this project's folder. For instance you can run, 

`python -m soccer_twos.watch -m1 example_player_agent -m2 ceia_baseline_agent`

, to examine the random agent vs. the baseline agent.


## PACE commands
```bash
salloc -N1 -t2:00:00 --gres=gpu:V100:1 --cpus-per-task=16 --mem=128G
```
