# SRL-MBPO-MPS
- 强化学习框架: [Force](https://github.com/gwthomas/force).

## 环境需求
- mujuco210
- python==3.8
- numpy==1.24.2
- matplotlib==3.7.0
- tqdm==4.64.1
- h5py==3.8.0
- opencv-python==4.7.0.68
- torch==1.5.0
- torchvision==0.6.0
- gym==0.17.2
- mujoco-py==2.1.2.14

## 参数设置
- 在`src/defaults.py`中设置`ROOT_DIR`, 用以保存数据.
- 在`src/env/torch_wrapper.py`中设置`ACTION_SIGMA`, 并根据当前任务设置动作的上下限`clip_sto_action`.


## 运行
在根目录下运行:

    python main.py -c config/ENV.json

其中, ENV是当前要选择的任务环境. 另外,可以通过命令行重写超参数, 详见`src/cli.py`.

`render.py`用来渲染动画,须设置保存训练数据的路径名和文件名.