# Mosquito classification API

This serves the Humbug Mosquito classifier using TorchServe

## Requirements
```
pip install torchserve torch-model-archiver torch-workflow-archiver captum nvgpu
```

## Audio processing requirements
```
pip install librosa
```

## Creating an archive


### MED
```
torch-model-archiver --model-name midsmedmodel --version 1.0 --model-file src/med/mids_med.py --serialized-file ../../models/model_presentation_draft_2022_04_07_11_52_08.pth --handler src/med/model_handler.py
```

### MSC
```
torch-model-archiver --model-name midsmscmodel --version 1.0 --model-file src/msc/mids_msc.py --serialized-file ../../models/model_e186_2022_10_11_11_18_50.pth --handler src/msc/model_handler.py --extra-files src/msc/index_to_name.json
```


Replace the path to the checkpoint as needed

## Serving a model
Move the model archive into the serving directory if not already present

#### MSC
```
mv midsmscmodel.mar models/
```

#### MED
```
mv midsmedmodel.mar models/
```

Start the server and load all the models
```
torchserve --start --model-store ./models --models all
```

Registering the models:
> this only needs to be done the first time you're using the server or models.  It will remember the models on subsequent startups

```
curl -X POST  "http://localhost:8081/models?url=mids_med/midsmedmodel.mar"
curl -X POST  "http://localhost:8081/models?url=mids_msc/midsmscmodel.mar"
```

Add Workers:
> This only needs to be done the first time you're using the server or models.  It will remember the worker settings on subsequent startups.

```
curl -v -X PUT "http://localhost:8081/models/midsmedmodel?min_worker=1"
curl -v -X PUT "http://localhost:8081/models/midsmscmodel?min_worker=1"
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

## Send async requests for an entire file
```
import aiohttp
import asyncio

predictions = {}
async def main():
    sample_length = int(config.min_duration * config.rate)
    async with aiohttp.ClientSession() as session:
        for start in tqdm(range(0, signal.shape[0], sample_length)):
            msc_url = "http://localhost:8080/predictions/midsmscmodel"
            signal_window = signal[start:start + sample_length].tobytes()
            async with session.post(msc_url, data=signal_window) as resp:
                model_out = await resp.text()
                bapredictionstch[start] = json.loads(model_out)

asyncio.run(main())
```

Result will be a dictionary of offsets:

```
{0: {'an funestus ss': 0.44641974568367004,
  'culex pipiens complex': 0.3559141457080841,
  'an arabiensis': 0.19581685960292816,
  'an squamosus': 0.0006995390285737813,
  'ma uniformis': 0.0005431393510662019,
  'ma africanus': 0.00040533419814892113,
  'an coustani': 0.00013025043881498277,
  'ae aegypti': 7.09825981175527e-05},
 15360: {'an funestus ss': 0.9878707528114319,
  'an arabiensis': 0.007386358920484781,
  'culex pipiens complex': 0.004397190175950527,
  'ma africanus': 0.00013819077867083251,
  'an squamosus': 9.089102240977809e-05,
  'ma uniformis': 5.5244716349989176e-05,
  'an coustani': 4.7722263843752444e-05,
  'ae aegypti': 1.377329772367375e-05},
 30720: {'an funestus ss': 0.9979367256164551,
  'culex pipiens complex': 0.0019879990722984076,
  'an arabiensis': 6.328509334707633e-05,
  'ma africanus': 4.563854417938273e-06,
  'an coustani': 4.050409643241437e-06,
  'an squamosus': 1.6617236724414397e-06,
  'ma uniformis': 1.331052203568106e-06,
  'ae aegypti': 4.1433537489865557e-07},
 46080: {'an funestus ss': 0.9640965461730957,
  'an arabiensis': 0.028270814567804337,
  'culex pipiens complex': 0.006991065572947264,
  'an squamosus': 0.00021047884365543723,
  'ma uniformis': 0.00018406323215458542,
  'an coustani': 0.0001297990675084293,
  'ma africanus': 9.884824248729274e-05,
  'ae aegypti': 1.8288870705873705e-05},
 61440: {'culex pipiens complex': 0.8509525656700134,
  'an arabiensis': 0.08412139862775803,
  'an funestus ss': 0.06323976069688797,
  'an squamosus': 0.0005122318980284035,
  'ma uniformis': 0.00046429975191131234,
  'ma africanus': 0.0003711211320478469,
  'an coustani': 0.00024704699171707034,
  'ae aegypti': 9.155666339211166e-05},
 76800: {'an funestus ss': 0.9954197406768799,
  'an arabiensis': 0.0029900874942541122,
  'culex pipiens complex': 0.0013661232078447938,
  'an squamosus': 0.00011221396562177688,
  'ma africanus': 4.244718002155423e-05,
  'ma uniformis': 4.197566522634588e-05,
  'an coustani': 2.3843593226047233e-05,
  'ae aegypti': 3.503900188661646e-06}
```

# Prediction script sequence to build a study
0. Ensure TorchServe is running as per the above info
1. Create a CSV file that contains the study uuids. e.g. `study_uuids.csv` and a CSV file for the study dates e.g. `study_dates.csv`.
2. Run the database extractor with the date range and path to recordings to create a job input CSV for MED.
```
python3 src/predict/extract_from_db.py examples/study/study_uuids.csv examples/study/study_dates.csv /home/deveshk/dbmount/MozzWear
```
3. Run the MED predict script using the CSV file from the previous step as input as well as a destination directory.
```
python3 src/predict/predict_med.py --csv MED_Task_2021-02-02_to_2022-02-02.csv --dst /humbug-data/study_1_med
```
4. RUN the MSC predict script on the output folder from MED
```
python3 src/predict/predict_msc.py /humbug-data/study_1_med /humbug-data/study_1_msc
```
* The MSC destination folder will contain the clipped WAV files, the audacity labels, and CSVs will all of the experiment metrics.