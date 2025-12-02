# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

You can find the original [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

> Original author: *Hongrui Zheng*

> Author: *Hyeokjin Kwon, Geunje Cheon, Junseok Kim*

<!-- ## Install Docker
If you are not using Ubuntu 18.04, you have to use Docker to install Ubuntu 18.04.
If you are, skip this section.

1. Update
```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
```

2. Repository key
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

3. Add apt-repository
```
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

4. Install recent Docker
```
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
```

5. Check docker system
```
sudo systemctl status docker

# ● docker.service - Docker Application Container Engine
#      Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset>
#      Active: active (running) since Sat 2024-05-25 18:21:35 KST; 3min 23s ago
# TriggeredBy: ● docker.socket
#        Docs: https://docs.docker.com
# ...
```

6. Make Super USER to not use sudo every time
```
sudo usermod -aG docker $USER
```
- After this, you should relogin.

7. Check example container
```
docker container run hello-world -->
<!-- ``` -->


## Quickstart
We recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
cd [your-virtualenv-folder]
virtualenv F1tenth --python=python3.8
source [your-virtualenv-folder]/F1tenth/bin/activate
# pip install poetry
pip install --upgrade pip

cd [your-workspace-folder]
git clone https://github.com/rllab-snu/Autonomous-Driving-Student
cd installation
pip install -e .
cd [your-workspace-folder]
pip install -r requirements.txt
```

Then you can run a quick waypoint follow example by:
```bash
cd installation/examples
python3 waypoint_follow.py --render --map Spielberg
```

## Calibration
After you complete the calibration process on the real RC car, the next step is **to ensure that the configuration settings in your simulation environment match those of the physical RC car**.

This alignment is crucial because it allows the simulation to accurately replicate the behavior of the real car, ensuring that the virtual tests and scenarios you run will closely mirror what would happen in the real world.

```bash
# calculate current max steering radius
cd installation/examples
python3 run_in_circle.py --render
```

You can change ***max_steering*** in *'configs/task/f1tenth.yaml'*.

## Map
There are 2 ways to use maps: 
1. random generate maps 
2. use f1 tracks

### Random Generation
```bash
cd [your-workspace-folder]
cd installation/examples

# you need to use different seed to make different maps.
python3 random_trackgen.py --seed 123 --name your_map_name

# or you can generate with your own points
python3 random_trackgen.py --checkpoints test.csv
```

### F1 Tracks
Use downloaded f1 tracks.
```python
MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin', 
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang', 
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya', 
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']
```

## Training
### Imitation Learning
You can collect expert data from PID controller.
```bash
python3 main_il.py --name imitation --algo_idx 1 --wandb  # --save_video or --live_video
```


### Reinforcement Learning
You can train your own models using RL algorithms : PPO, SAC.
```bash
python3 main_rl.py --algo ppo --name MYPPO --algo_idx 1 --wandb # --save_video or --live_video
```

### Test
You can test your model with evaluation code.
```bash
python3 main_rl.py --test --algo ppo --name MYPPO --algo_idx 1 --model_num 10000 # --save_video or --live_video
```

### Visualize
You can visualize your model's performance with loggers.
```python
def main():
        ...

        item_list = ['score']   # or other metrics
        algo_list.append({
        'name': 'ppo',
        'logs': [f'results/MYPPO/{i}' for i in range(1,4)]
        })
        algo_list.append({
        'name': 'sac',
        'logs': [f'results/MYSAC/{i}' for i in range(1,4)]
        })

        ...
```
```bash
python3 visualize.py
```

## Leaderboard
After finish your model, you can check success ratio and laptimes using your best model for sim2real.
```bash
python3 leaderboard.py --algo_name MYPPO --algo_idx 1 # --save_video or --live_video
```
