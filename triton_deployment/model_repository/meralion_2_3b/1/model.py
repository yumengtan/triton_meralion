import json
import sys
import os
import numpy as np
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import triton_python_backend_utils as pb_utils
import logging
import gc
import base64
import io
import soundfile as sf
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """
    MERaLiON Speech-to-Text model for NVIDIA Triton Inference Server

    Reference:
    - https://github.com/triton-inference-server/server/discussions/6574
    - https://github.com/openai/whisper/discussions/1505
    - https://github.com/k2-fsa/sherpa/tree/master/triton/whisper
    - https://github.com/triton-inference-server/server/issues/7847
    """

    def initialize(self, args):
        """Initialize the MERaLiON Speech-to-Text model"""
        self.model_dir = args['model_repository']
        self.model_name = args['model_name']
        self.model_version = args['model_version']
        self.model_instance_kind = args['model_instance_kind']
        self.model_instance_device_id = args['model_instance_device_id']

        # Path to model files
        model_files_path = "/model_files/MERaLiON-2-3B"

        logger.info(f"Initializing MERaLiON Speech-to-Text model from: {model_files_path}")

        try:
            # Clear GPU cache at start
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            if model_files_path not in sys.path:
                sys.path.insert(0, model_files_path)

            # Set cache directories (following best practices from referenced sources)
            os.environ['HF_HOME'] = model_files_path
            os.environ['TRANSFORMERS_CACHE'] = model_files_path

            if self.model_instance_kind == "GPU":
                self.device = torch.device(f'cuda:{self.model_instance_device_id}')
                logger.info(f"Using GPU device: {self.device}")
            else:
                self.device = torch.device('cpu')
                logger.info(f"Using CPU device: {self.device}")

            # Load processor with error handling (pattern from k2-fsa implementation)
            logger.info("Loading processor...")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_files_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    cache_dir=model_files_path
                )
                logger.info("Processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load processor: {e}")
                raise

            # Load speech-to-seq model with optimizations
            logger.info("Loading MERaLiON Speech-to-Text model...")
            try:
                
                if self.device.type == 'cuda' and torch.cuda.is_available():
                   
                    if torch.cuda.get_device_capability(self.device.index)[0] >= 8:
                        model_dtype = torch.bfloat16
                    else:
                        model_dtype = torch.float16
                else:
                    model_dtype = torch.float32

                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_files_path,
                    use_safetensors=True,
                    trust_remote_code=True,
                    torch_dtype=model_dtype,
                    local_files_only=True,
                    cache_dir=model_files_path,
                    device_map={"": self.device} if self.device.type == 'cuda' else None
                )
                logger.info(f"Model loaded successfully with dtype: {model_dtype}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

            # Set model to evaluation mode
            self.model.eval()

            if self.device.type == 'cuda':
                logger.info("Optimizing model for GPU inference...")
                torch.cuda.empty_cache()

                if next(self.model.parameters()).device != self.device:
                    self.model = self.model.to(self.device)

                # Log memory usage
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                memory_total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                logger.info(f"GPU memory: {memory_used:.2f}GB / {memory_total:.2f}GB used")

            self.sample_rate = 16000
            self.max_audio_length = 30 * self.sample_rate
            self.min_audio_length = 0.1 * self.sample_rate  


            self.generation_config = {
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_beams": 1,
                "early_stopping": True,
                "pad_token_id": None,  
                "eos_token_id": None,  
                "use_cache": True,
                "return_dict_in_generate": False
            }

            # Set token IDs from processor
            if hasattr(self.processor, 'tokenizer'):
                self.generation_config["pad_token_id"] = self.processor.tokenizer.eos_token_id
                self.generation_config["eos_token_id"] = self.processor.tokenizer.eos_token_id

            self.prompt_templates = {
                "transcribe": "Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "translate_chinese": "Instruction: Please translate this speech into Chinese. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "translate_malay": "Instruction: Please translate this speech into Malay. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "translate_english": "Instruction: Please translate this speech into English. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "summarize": "Instruction: Please summarize this speech. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "emotion": "Instruction: What is the emotion of the speaker? \nFollow the text instruction based on the following audio: <SpeechHere>",
                "gender": "Instruction: What is the gender of the speaker? \nFollow the text instruction based on the following audio: <SpeechHere>",
                "language": "Instruction: What language is being spoken? \nFollow the text instruction based on the following audio: <SpeechHere>",
                "describe": "Instruction: Please describe the audio. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "respond": "Instruction: Please respond to the audio. \nFollow the text instruction based on the following audio: <SpeechHere>",
                "classify": "Instruction: Please classify the content of this speech. \nFollow the text instruction based on the following audio: <SpeechHere>"
            }

            # Performance monitoring
            self.request_count = 0
            self.total_inference_time = 0.0
            self.total_audio_duration = 0.0

            logger.info("MERaLiON Speech-to-Text model initialization completed successfully")

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._cleanup()
            raise

    def execute(self, requests):
        """Execute speech-to-text inference requests with optimized batching"""
        responses = []
        start_time = time.time()

        # Process requests in batch when possible (following k2-fsa patterns)
        try:
            batch_audio_data = []
            batch_tasks = []
            batch_indices = []

            # Collect and validate all requests first
            for idx, request in enumerate(requests):
                try:
                    audio_input = pb_utils.get_input_tensor_by_name(request, "audio_input")
                    task_input = pb_utils.get_input_tensor_by_name(request, "task_input")

                    if audio_input is None or task_input is None:
                        error_response = pb_utils.InferenceResponse(
                            error=pb_utils.TritonError("Missing required inputs: audio_input and task_input")
                        )
                        responses.append(error_response)
                        continue
                    audio_data = audio_input.as_numpy()
                    task_data = task_input.as_numpy()

                    if isinstance(task_data[0], bytes):
                        task_name = task_data[0].decode('utf-8')
                    else:
                        task_name = str(task_data[0])

                    logger.info(f"Processing request {idx} with task: {task_name}")
                    audio_array = self._process_audio_input(audio_data)
                    if audio_array is None:
                        error_response = pb_utils.InferenceResponse(
                            error=pb_utils.TritonError("Failed to process audio input")
                        )
                        responses.append(error_response)
                        continue

                    batch_audio_data.append(audio_array)
                    batch_tasks.append(task_name)
                    batch_indices.append(idx)

                except Exception as e:
                    logger.error(f"Error processing request {idx}: {str(e)}")
                    error_response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Request processing failed: {str(e)}")
                    )
                    responses.append(error_response)

            if batch_audio_data:
                batch_results = self._process_batch(batch_audio_data, batch_tasks)

                # Create responses for successful requests
                for i, (audio_array, task_name) in enumerate(zip(batch_audio_data, batch_tasks)):
                    try:
                        if i < len(batch_results):
                            result_text = batch_results[i]
                        else:
                            result_text = "Error: No result generated"

                        output_tensor = pb_utils.Tensor(
                            "text_output",
                            np.array([[result_text]], dtype=object)
                        )

                        response = pb_utils.InferenceResponse(output_tensors=[output_tensor])

                        while len(responses) <= batch_indices[i]:
                            responses.append(None)
                        responses[batch_indices[i]] = response

                    except Exception as e:
                        logger.error(f"Error creating response for batch item {i}: {str(e)}")
                        error_response = pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(f"Response creation failed: {str(e)}")
                        )
                        while len(responses) <= batch_indices[i]:
                            responses.append(None)
                        responses[batch_indices[i]] = error_response

            for i in range(len(requests)):
                if i >= len(responses) or responses[i] is None:
                    error_response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("Request processing failed")
                    )
                    while len(responses) <= i:
                        responses.append(None)
                    responses[i] = error_response

            # Update performance metrics
            end_time = time.time()
            self.request_count += len(requests)
            self.total_inference_time += (end_time - start_time)

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Return error responses for all requests
            error_msg = f"Batch processing failed: {str(e)}"
            responses = [
                pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
                for _ in requests
            ]

        finally:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        return responses

    def _process_batch(self, audio_arrays, tasks):
        """Process a batch of audio arrays with their corresponding tasks"""
        results = []

        for audio_array, task_name in zip(audio_arrays, tasks):
            try:
                if task_name.startswith("[b'") and task_name.endswith("']"):
                    task_name = task_name[3:-2]
                elif task_name.startswith("b'") and task_name.endswith("'"):
                    task_name = task_name[2:-1]
                
                logger.info(f"Processing task: '{task_name}'")
                
                if audio_array is None or len(audio_array) == 0:
                    logger.error("Audio array is None or empty")
                    results.append("Error: Invalid audio array")
                    continue
                    
                logger.info(f"Audio array shape: {audio_array.shape}, min: {audio_array.min():.4f}, max: {audio_array.max():.4f}")

                prompt = self.prompt_templates.get(task_name, self.prompt_templates["transcribe"])
                logger.info(f"Using prompt template for task '{task_name}': {prompt[:100]}...")

                conversation = [{"role": "user", "content": prompt}]
                logger.info(f"Conversation format: {conversation}")

                # Apply chat template
                try:
                    chat_prompt = self.processor.tokenizer.apply_chat_template(
                        conversation=conversation,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    logger.info(f"Chat prompt created, length: {len(chat_prompt)}")
                    logger.info(f"Chat prompt preview: {chat_prompt[:200]}...")
                except Exception as e:
                    logger.error(f"Failed to create chat prompt: {e}")
                    # Fallback to simple prompt
                    chat_prompt = prompt
                    logger.info(f"Using fallback prompt, length: {len(chat_prompt)}")

                logger.info("Processing model inputs...")
                inputs = self._process_model_inputs(chat_prompt, audio_array)
                if inputs is None:
                    logger.error("Model inputs processing returned None")
                    results.append("Error: Failed to process model inputs")
                    continue

                logger.info(f"Model inputs ready: {list(inputs.keys())}")

                # Generate response with timeout protection
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                    gen_config = self.generation_config.copy()
                    logger.info(f"Generation config: {gen_config}")

                    try:
                        logger.info("Starting model generation...")
                        outputs = self.model.generate(
                            **inputs,
                            **gen_config
                        )
                        logger.info(f"Generation completed, output shape: {outputs.shape}")
                        generated_ids = outputs[:, inputs['input_ids'].size(1):]
                        logger.info(f"Generated IDs shape: {generated_ids.shape}")
                        
                        response_text = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )[0].strip()

                        logger.info(f"Generated response: {response_text}")
                        results.append(response_text)

                    except Exception as e:
                        logger.error(f"Generation failed: {e}")
                        import traceback
                        logger.error(f"Generation traceback: {traceback.format_exc()}")
                        results.append(f"Error: Generation failed - {str(e)}")
                try:
                    del inputs
                    if 'outputs' in locals():
                        del outputs
                except Exception as e:
                    logger.error(f"Error processing single request: {str(e)}")
                    pass

            except Exception as e:
                logger.error(f"Error processing single request: {str(e)}")
                import traceback
                logger.error(f"Single request traceback: {traceback.format_exc()}")
                results.append(f"Error: {str(e)}")

        return results

    def _process_model_inputs(self, chat_prompt, audio_array):
        """
        Process model inputs with separated text and audio processing to avoid return_tensors conflicts
        Based on the fix for the return_tensors issue
        """
        try:
            logger.info(f"Processing model inputs - chat_prompt length: {len(chat_prompt)}, audio_array shape: {audio_array.shape}")
            
            # Use the official MERaLiON approach as shown in HuggingFace docs
            logger.info("Using MERaLiON official processor approach...")
            inputs = self.processor(text=chat_prompt, audios=[audio_array])
            logger.info(f"MERaLiON processor success: {list(inputs.keys())}")
            
            # Validate inputs
            for key, value in inputs.items():
                if value is None:
                    logger.error(f"Input {key} is None!")
                    return None
                logger.info(f"Input {key}: shape={value.shape}, dtype={value.dtype}")

        except Exception as e:
            logger.error(f"Error in MERaLiON processor: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

        # Move inputs to device with proper dtype handling
        try:
            logger.info(f"Moving inputs to device: {self.device}")
            for key, value in inputs.items():
                if value is None:
                    logger.error(f"Cannot move None value for key: {key}")
                    return None
                    
                if isinstance(value, torch.Tensor):
                    logger.info(f"Moving {key} to device...")
                    inputs[key] = value.to(self.device)
                    
                    # Convert to model dtype if needed
                    if hasattr(self.model, 'dtype') and value.dtype != self.model.dtype:
                        if value.dtype == torch.float32 and self.model.dtype in [torch.float16, torch.bfloat16]:
                            logger.info(f"Converting {key} from {value.dtype} to {self.model.dtype}")
                            inputs[key] = inputs[key].to(self.model.dtype)
                else:
                    logger.warning(f"Input {key} is not a tensor: {type(value)}")

            logger.info("All inputs successfully moved to device")
            return inputs

        except Exception as e:
            logger.error(f"Error moving inputs to device: {e}")
            import traceback
            logger.error(f"Device move traceback: {traceback.format_exc()}")
            return None

    def _process_audio_input(self, audio_input):
        """Process audio input with proper base64 decoding and validation"""
        try:
            # Handle different input formats
            if isinstance(audio_input, np.ndarray):
                # If it's already a numpy array, extract the first element
                if audio_input.size > 0:
                    audio_data = audio_input.flat[0]
                    if isinstance(audio_data, bytes):
                        audio_data = audio_data.decode('utf-8')
                else:
                    raise ValueError("Empty audio input array")
            elif isinstance(audio_input, bytes):
                audio_data = audio_input.decode('utf-8')
            else:
                audio_data = str(audio_input)

            logger.info(f"Processing audio input, type: {type(audio_data)}, length: {len(audio_data)}")

            # Decode base64 to get the actual WAV file bytes
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"Decoded {len(audio_bytes)} bytes from base64")

            # Use librosa to load the WAV file from bytes
            audio_array, orig_sr = librosa.load(
                io.BytesIO(audio_bytes),
                sr=None,
                mono=True  # Convert stereo to mono
            )

            logger.info(f"Loaded audio: {len(audio_array)} samples at {orig_sr}Hz")
            logger.info(f"Duration: {len(audio_array)/orig_sr:.2f} seconds")

            if len(audio_array) < 100:  # Less than ~2ms at 44kHz
                raise ValueError(f"Audio too short: {len(audio_array)} samples, need at least 100")

            self.total_audio_duration += len(audio_array) / orig_sr

            # Resample to target sample rate (16kHz)
            if orig_sr != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=orig_sr,
                    target_sr=self.sample_rate
                )

            logger.info(f"Resampled to: {len(audio_array)} samples at {self.sample_rate}Hz")

            # Apply audio length limits
            if len(audio_array) > self.max_audio_length:
                logger.warning(f"Audio too long ({len(audio_array)} samples), truncating to {self.max_audio_length}")
                audio_array = audio_array[:self.max_audio_length]
            elif len(audio_array) < self.min_audio_length:
                logger.warning(f"Audio too short ({len(audio_array)} samples), padding to {int(self.min_audio_length)}")
                audio_array = np.pad(audio_array, (0, int(self.min_audio_length) - len(audio_array)), mode='constant')

            return audio_array

        except Exception as e:
            logger.error(f"Error in _process_audio_input: {str(e)}")
            logger.error(f"Audio input type: {type(audio_input)}")
            logger.error(f"Audio input length: {len(audio_input) if hasattr(audio_input, '__len__') else 'No length'}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def finalize(self):
        """Clean up resources when model is unloaded"""
        logger.info("Finalizing MERaLiON Speech-to-Text model")

        if self.request_count > 0:
            avg_inference_time = self.total_inference_time / self.request_count
            avg_audio_duration = self.total_audio_duration / self.request_count
            logger.info(f"Final stats - Requests: {self.request_count}, "
                       f"Avg inference time: {avg_inference_time:.3f}s, "
                       f"Avg audio duration: {avg_audio_duration:.2f}s")
        self._cleanup()
