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


## picamera2 installation
```
sudo apt install -y python3-picamera2
pip3 install picamera2
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
sudo python3 -m pip install rpyc --break-system-packages
```

Then we can run the server
```
sudo python3 src/server/server.py
```
