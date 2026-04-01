conda create --name soccertwos python=3.8 -y
conda activate soccertwos
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install torch==1.13.1
pip install aiohttp==3.7.4 aioredis==1.3.1 dm-tree==0.1.6 gym==0.19.0 gym-unity==0.27.0 numpy==1.23.5
pip install mlagents==0.27.0 mlagents-envs==0.27.0 --no-deps
conda install grpcio=1.43.0 -c conda-forge
conda install pip=23.3.2 -c conda-forge # `pip -V` and `python -m pip --version` to be 23.3.2
pip install ray==1.13.0 "ray[tune]==1.13.0" "ray[rllib]==1.13.0"
pip install soccer-twos==0.1.14 --no-deps

python3 -c "
import soccer_twos, os
pkg_path = os.path.dirname(soccer_twos.__file__)
content = open(os.path.join(pkg_path, 'package.py')).read()
content = content.replace(
    'if not Path(TRAINING_ENV_PATH).is_file() and not Path(ROLLOUT_ENV_PATH).is_file():',
    'if not Path(TRAINING_ENV_PATH + \".app\").is_dir() and not Path(ROLLOUT_ENV_PATH + \".app\").is_dir():'
)
content = content.replace(
    'TRAINING_ENV_PATH = \"mac_os/soccer-twos.app/Contents/MacOS/UnityEnvironment\"',
    'TRAINING_ENV_PATH = \"mac_os/soccer-twos\"'
)
content = content.replace(
    'ROLLOUT_ENV_PATH = \"mac_os/watch-soccer-twos.app/Contents/MacOS/UnityEnvironment\"',
    'ROLLOUT_ENV_PATH = \"mac_os/watch-soccer-twos\"'
)
open(os.path.join(pkg_path, 'package.py'), 'w').write(content)
print('done')
"