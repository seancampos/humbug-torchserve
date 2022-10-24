# Mosquito classification API

This serves the Humbug Mosquito classifier using TorchServe

## Requirements
```
pip install torchserve torch-model-archiver torch-workflow-archiver
```

## Creating an archive
```
torch-model-archiver --model-name midsmscmodel --version 1.0 --model-file src/msc/mids_msc.py --serialized-file ../../models/model_e186_2022_10_11_11_18_50.pth --handler src/msc/model_handler.py --extra-files src/msc/index_to_name.json
```
Replace the path to the checkpoint as needed

## Serving a model
Move the model archive into the serving directory if not already present
```
mv midsmscmodel.mar models/
```
Start the server and load all the models
```
torchserve --start --model-store ./models --models all
```

## Stopping the server
```
torchserve --stop
```


## Making requests
The service expects a numpy array of of length 15360 (1.92 seconds * 8000 bit rate).
```
res = requests.post("http://localhost:8080/predictions/midsmscmodel",
                    data=waveform[0:int(config.min_duration * config.rate)].tobytes())
```
The response is a JSON object with the species and probabilities
```
{'an funestus ss': 0.3590160310268402,
 'culex pipiens complex': 0.3258729875087738,
 'an arabiensis': 0.3121899366378784,
 'an squamosus': 0.0009254553588107228,
 'ma africanus': 0.0008028328302316368,
 'an coustani': 0.0005933775100857019,
 'ma uniformis': 0.0004665222077164799,
 'ae aegypti': 0.00013280179700814188}
 ```
