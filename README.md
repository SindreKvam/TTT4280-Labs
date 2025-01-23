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

