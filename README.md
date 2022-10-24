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
Start the server
```
torchserve --start --model-store ./models
```

