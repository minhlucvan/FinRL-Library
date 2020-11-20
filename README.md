# FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
This repository refers to the codes for our [paper]() that appears at Deep RL Workshop, NeurIPS 2020


## Abstract
As deep reinforcement learning (DRL) has been recognized as an effective approach in quantitative finance, getting hands-on experiences is attractive to begin-ners. However, to train a practical DRL trading agent that decides where to trade, at what price, and what quantity involves error-prone and arduous developmentand debugging. 

In this paper, we introduce a DRL library FinRL that facilitates beginners to expose themselves to quantitative finance and to develop their own stock trading strategies. Along with easily-reproducible tutorials, FinRL library allows users to streamline their own developments and to compare with existing schemes easily.  Within FinRL, virtual environments are configured with stockmarket datasets, trading agents are trained with neural networks, and extensive backtesting is analyzed via trading performance. Moreover, it incorporates important trading constraints such as transaction cost, market liquidity and the investor’s degree of risk-aversion. 

FinRL is featured with completeness, hands-on tutorial and reproducibility that favors beginners: (i) at multiple levels of time granularity, FinRL simulates trading environments across various stock markets, including NASDAQ-100, DJIA, S&P 500, HSI, SSE 50, and CSI 300; (ii) organized in a layered architecture with modular structure, FinRL provides fine-tuned state-of-the-art DRL algorithms (DQN, DDPG, PPO, SAC, A2C, TD3, etc.), commonly-usedreward functions and standard evaluation baselines to alleviate the debugging work-loads and promote the reproducibility, and (iii) being highly extendable, FinRLreserves a complete set of user-import interfaces. 

Furthermore, we incorporated three application demonstrations, namely single stock trading, multiple stock trading, and portfolio allocation. 

<img src=figs/stock_trading.png width="600">

## Reference
Xiao-Yang Liu, Hongyang Yang, Shan Zhong, and Anwar Walid. 2020. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. In ICAIF ’20: ACM International Conference on AI in Finance, Oct. 15–16, 2020, Manhattan, NY. ACM, New York, NY, USA.

## [Our Medium Blog]()
## Installation:

Clone this repository
```shell
git clone https://github.com/AI4Finance-LLC/FinRL-Library.git
```

Install the unstable development version of FinRL:
```shell
pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git
```



### Prerequisites
For [OpenAI Baselines](https://github.com/openai/baselines), you'll need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).
    
### Create and Activate Virtual Environment (Optional but highly recommended)
cd into this repository
```bash
cd Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
```
Under folder /Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020, create a virtual environment
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages. 

**Virtualenvs can also avoid packages conflicts.**

Create a virtualenv **venv** under folder /Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020
```bash
virtualenv -p python3 venv
```
To activate a virtualenv:
```
source venv/bin/activate
```

## Dependencies

The script has been tested running under **Python >= 3.6.0**, with the folowing packages installed:

```shell
pip install -r requirements.txt
```

### Questions

### About Tensorflow 2.0: https://github.com/hill-a/stable-baselines/issues/366

If you have questions regarding TensorFlow, note that tensorflow 2.0 is not compatible now, you may use

```bash
pip install tensorflow==1.15.4
 ```

If you have questions regarding Stable-baselines package, please refer to [Stable-baselines installation guide](https://github.com/hill-a/stable-baselines). Install the Stable Baselines package using pip:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Run 
```shell
python main.py
```
## Backtesting

Use Quantopian's [pyfolio package](https://github.com/quantopian/pyfolio) to do the backtesting.

[Backtesting script]()

## Status

<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 1.0.1
	Changes: added ensemble strategy
* 0.0.1
    Simple version
</div>
</details>

## Data
The stock data we use is pulled from Yahoo Finance API
<img src=figs/data.PNG width="500">


## Performance
<img src=figs/performance.png>
