# README

To pull submodules run the following command
```
git pull --recurse-submodules
```

## pigpio installation
```
cd pigpio
make
sudo make install
```


## adc_sampler

To compile, run the following commands:
```
cd src
make
```

Usage is as following:
```
sudo adc_sampler <sample-count> [output]
```

## Run RPyC server
The RPyC server is required to be ran by root to allow using DMAs.

First we need to install python packages on root level:
```
sudo apt install python3-numpy
sudo python -m pip install rpyc --break-system-packages
```

Then we can run the server
```
sudo python src/server/server.py
```
