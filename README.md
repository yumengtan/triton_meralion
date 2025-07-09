# Triton Deployment for MeraLion2-3B

This repository contains the configuration and deployment files for running the MeraLion2-3B model using NVIDIA Triton Inference Server.

## File Structure

```
triton_deployment/
├── Dockerfile
├── models/
│   └── MERaLiON-2-3B/
└── model_repository/
    └── meralion_2_3b/
        ├── config.pbtxt
        └── 1/
            └── model.py
```

## Directory Overview

- **Dockerfile**: Container configuration for the Triton server environment
- **models/MERaLiON-2-3B/**: Directory containing the actual model files and weights
- **model_repository/meralion_2_3b/**: Triton model repository structure
  - **config.pbtxt**: Model configuration file defining input/output specifications
  - **1/**: Model version directory (version 1)
    - **model.py**: Python backend implementation for the model

## Prerequisites

- Docker with GPU support
- NVIDIA Container Toolkit
- Sufficient GPU memory for the MeraLion2-3B model

## Deployment

### Building the Container

Build the Docker image from the repository root:

```bash
docker build -t triton_server_2:latest .
```

### Running the Server

Execute the following command to start the Triton server:

```bash
docker run --gpus all -it   --shm-size=2G   --ulimit memlock=-1   --ulimit stack=67108864   -e HF_HUB_OFFLINE=1   -e TRANSFORMERS_OFFLINE=1   -e HF_DATASETS_OFFLINE=1   -v ${PWD}/model_repository:/models   -v ${PWD}/models:/model_files   -p 8000:8000 -p 8001:8001 -p 8002:8002   trition_server   bash -c "pip install librosa soundfile transformers==4.50.1 && tritonserver --model-repository=/models --exit-on-error=false"
```

### Command Parameters Explained

- `--gpus all`: Enables access to all available GPUs
- `--net=host`: Uses host networking for optimal performance
- `--shm-size=2G`: Allocates 2GB shared memory
- `--ulimit memlock=-1`: Removes memory lock limits
- `--ulimit stack=67108864`: Sets stack size limit
- `-e HF_HUB_OFFLINE=1`: Disables Hugging Face Hub access
- `-e TRANSFORMERS_OFFLINE=1`: Runs transformers in offline mode
- `-e HF_DATASETS_OFFLINE=1`: Disables datasets online access
- `-v ${PWD}/model_repository:/models`: Mounts model repository
- `-v ${PWD}/models:/model_files`: Mounts model files directory
- `--model-repository=/models`: Specifies Triton model repository path
- `--log-verbose=1`: Enables verbose logging

## Setting up the Model Repository

A model repository is Triton's way of reading your models and any associated metadata with each model (configurations, version files, etc.). These model repositories can live in a local or network attached filesystem, or in a cloud object store like AWS S3, Azure Blob Storage or Google Cloud Storage. For more details on model repository location, refer to the documentation. Servers can use also multiple different local repositories.

The model repository follows a specific structure format:

```
<model-repository>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
```

## Adding a New Model

To add a new model to the repository:

1. **Create the model directory structure**:
   ```bash
   mkdir -p model_repository/<new-model-name>/1
   ```

2. **Add the model configuration file** (`config.pbtxt`):
   ```
   name: "<new-model-name>"
   platform: "python"
   max_batch_size: 1
   
   input [
     {
       name: "audio_input"
       data_type: TYPE_STRING
       dims: [ 1 ]
     },
     {
       name: "task_input"
       data_type: TYPE_STRING
       dims: [ 1 ]
     }
   ]
   
   output [
     {
       name: "text_output"
       data_type: TYPE_STRING
       dims: [ 1 ]
     }
   ]
   
   instance_group [
     {
       count: 1
       kind: KIND_GPU
     }
   ]
   ```

3. **Implement the model backend** (`model.py`):
   Create a Python file in the version directory (e.g., `1/model.py`) that implements the TritonPythonModel interface.

4. **Add model files**:
   Place your actual model weights and files in the `models/` directory and reference them in your `model.py` implementation.

## Configuration

The model configuration is defined in `config.pbtxt`. Ensure this file properly specifies:

- Model name and platform
- Input and output tensor specifications
- Maximum batch size
- Instance group configuration

## Model Loading

The model implementation in `model.py` handles:

- Model initialization and loading of MeraLion2-3B
- Audio preprocessing (base64 decoding, resampling to 16kHz)
- Speech-to-text inference with multiple task support
- Output postprocessing and response generation

### Available Tasks

The MeraLion2-3B model supports the following speech processing tasks:

- `transcribe` - Convert speech to text (default)
- `translate_chinese` - Translate speech to Chinese
- `translate_malay` - Translate speech to Malay  
- `translate_english` - Translate speech to English
- `summarize` - Summarize the speech content
- `emotion` - Detect speaker emotion
- `gender` - Identify speaker gender
- `language` - Identify the language being spoken
- `describe` - Describe the audio content
- `respond` - Generate a response to the audio
- `classify` - Classify the speech content

### Input Format

The model expects two inputs:

1. **audio_input** (TYPE_STRING): Base64-encoded WAV file
2. **task_input** (TYPE_STRING): Task name from the available tasks above

### Audio Requirements

- Format: WAV file (base64 encoded)
- Sample rate: Automatically resampled to 16kHz
- Duration: 0.1 to 30 seconds
- Channels: Automatically converted to mono

### Example Usage

To use the model, send requests either:
```
# Audio input: base64-encoded WAV file, describes about the audio file
curl -X POST http://localhost:8000/v2/models/meralion_2_3b/infer   -H "Content-Type: application/json"   -d @describe_request.json
```
or 
```
python3 generate_request.py --task transcribe --audio test.wav
```

## Verification

Once the server is running, you can verify the deployment by:

1. Checking server health: `curl -v localhost:8000/v2/health/ready`
2. Listing available models: `curl -v localhost:8000/v2/models`
3. Getting model metadata: `curl -v localhost:8000/v2/models/meralion_2_3b`

## Troubleshooting

- Ensure sufficient GPU memory is available
- Check that all model files are properly mounted
- Verify the config.pbtxt syntax matches Triton requirements
- Monitor container logs for initialization errors
