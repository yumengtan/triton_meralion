import base64
import json
import subprocess
import os
import time
import argparse

class AudioProcessor:
    def __init__(self, audio_file_path='test.wav', server_url='http://localhost:8000', 
                 model_name='meralion_2_3b', timeout=60, verbose=False, quiet=False):
        self.audio_file_path = audio_file_path
        self.server_url = server_url
        self.model_name = model_name
        self.timeout = timeout
        self.verbose = verbose
        self.quiet = quiet
        
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
        
        self.base64_audio = None
    
    def log(self, message, force=False):
        """Log message based on verbosity settings"""
        if force or (not self.quiet and (self.verbose or not message.startswith("Debug:"))):
            print(message)
    
    def error(self, message):
        """Always log error messages"""
        print(f"Error: {message}")
        
    def encode_audio(self):
        """Read and encode the audio file to base64"""
        try:
            with open(self.audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            self.base64_audio = base64.b64encode(audio_data).decode('utf-8')
            self.log(f"Base64 audio length: {len(self.base64_audio)} characters")
            return True
        except FileNotFoundError:
            self.error(f"Audio file '{self.audio_file_path}' not found")
            return False
        except Exception as e:
            self.error(f"Error encoding audio: {e}")
            return False
    
    def create_request(self, task):
        """Create a request JSON for a specific task"""
        if not self.base64_audio:
            self.error("Audio not encoded. Call encode_audio() first.")
            return None
            
        request = {
            "inputs": [
                {
                    "name": "audio_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [self.base64_audio]
                },
                {
                    "name": "task_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [task]
                }
            ]
        }
        return request
    
    def save_request_to_file(self, request, filename):
        """Save request to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(request, f, indent=2)
            return True
        except Exception as e:
            self.error(f"Error saving request to file: {e}")
            return False
    
    def send_curl_request(self, request_file, output_file=None):
        """Send curl request and return response"""
        url = f"{self.server_url}/v2/models/{self.model_name}/infer"
        
        curl_command = [
            'curl', '-X', 'POST', url,
            '-H', 'Content-Type: application/json',
            '-d', f'@{request_file}'
        ]
        
        self.log(f"Debug: Executing curl command: {' '.join(curl_command)}")
        
        try:
            result = subprocess.run(curl_command, capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode == 0:
                response_data = result.stdout
                
                # Save response to file if specified
                if output_file:
                    with open(output_file, 'w') as f:
                        f.write(response_data)
                
                return response_data
            else:
                self.error(f"Curl command failed with return code {result.returncode}")
                self.error(f"Error output: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.error(f"Request timed out after {self.timeout} seconds")
            return None
        except Exception as e:
            self.error(f"Error executing curl command: {e}")
            return None
    
    def parse_response(self, response_text):
        """Parse the JSON response and extract the result"""
        try:
            response_json = json.loads(response_text)
            
            # Extract the output (adjust based on your server's response format)
            if 'outputs' in response_json:
                outputs = response_json['outputs']
                if outputs and len(outputs) > 0:
                    output_data = outputs[0].get('data', [])
                    if output_data:
                        return output_data[0]
            
            return response_text  # Return raw response if parsing fails
        except json.JSONDecodeError:
            self.error("Failed to parse JSON response")
            return response_text
        except Exception as e:
            self.error(f"Error parsing response: {e}")
            return response_text
    
    def process_single_task(self, task, save_response=False, keep_request=False):
        """Process a single task and return the result"""
        self.log(f"\n--- Processing task: {task} ---", force=True)
        
        # Create request
        request = self.create_request(task)
        if not request:
            return None
        
        # Save request to file
        request_filename = f"request_{task}.json"
        if not self.save_request_to_file(request, request_filename):
            return None
        
        self.log(f"Created {request_filename}")
        
        # Send curl request
        response_filename = f"response_{task}.json" if save_response else None
        response = self.send_curl_request(request_filename, response_filename)
        
        if response:
            if save_response:
                self.log(f"Response saved to {response_filename}")
            
            parsed_result = self.parse_response(response)
            self.log(f"Result: {parsed_result}", force=True)
            
            # Clean up request file unless requested to keep
            if not keep_request:
                try:
                    os.remove(request_filename)
                except:
                    pass
            else:
                self.log(f"Request file kept: {request_filename}")
                
            return parsed_result
        else:
            self.error("Failed to get response")
            return None
    
    def process_all_tasks(self, delay_between_requests=1):
        """Process all tasks in the prompt templates"""
        if not self.encode_audio():
            return {}
        
        results = {}
        
        for task in self.prompt_templates.keys():
            result = self.process_single_task(task)
            results[task] = result
            
            # Add delay between requests to avoid overwhelming the server
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
        
        return results
    
    def process_specific_tasks(self, task_list, delay_between_requests=1):
        """Process only specific tasks"""
        if not self.encode_audio():
            return {}
        
        results = {}
        
        for task in task_list:
            if task in self.prompt_templates:
                result = self.process_single_task(task)
                results[task] = result
                
                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)
            else:
                print(f"Warning: Task '{task}' not found in prompt templates")
        
        return results
    
    def save_results_summary(self, results, filename="results_summary.json"):
        """Save all results to a summary file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results summary saved to {filename}")
        except Exception as e:
            print(f"Error saving results summary: {e}")

def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='Process audio files with various tasks using Triton Inference Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Transcribe audio
  python script.py --task transcribe --audio test.wav
  
  # Multiple tasks
  python script.py --task transcribe emotion language --audio test.wav
  
  # All tasks
  python script.py --all-tasks --audio test.wav
  
  # Custom server URL
  python script.py --task transcribe --audio test.wav --server-url http://192.168.1.100:8000
  
  # Save results with custom filename
  python script.py --task transcribe --audio test.wav --output results.json
        '''
    )
    
    # Audio file argument
    parser.add_argument(
        '--audio', '-a',
        required=True,
        help='Path to the audio file to process'
    )
    
    # Task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        '--task', '-t',
        nargs='+',
        choices=[
            'transcribe', 'translate_chinese', 'translate_malay', 'translate_english',
            'summarize', 'emotion', 'gender', 'language', 'describe', 'respond', 'classify'
        ],
        help='Task(s) to perform on the audio. Can specify multiple tasks.'
    )
    
    task_group.add_argument(
        '--all-tasks',
        action='store_true',
        help='Process all available tasks'
    )
    
    # Server configuration
    parser.add_argument(
        '--server-url', '-s',
        default='http://localhost:8000',
        help='Triton server URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--model-name', '-m',
        default='meralion_2_3b',
        help='Model name on Triton server (default: meralion_2_3b)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output', '-o',
        help='Output file for results summary (default: results_summary.json)'
    )
    
    parser.add_argument(
        '--save-responses',
        action='store_true',
        help='Save individual response files for each task'
    )
    
    parser.add_argument(
        '--keep-requests',
        action='store_true',
        help='Keep individual request JSON files (default: delete after use)'
    )
    
    # Processing options
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Request timeout in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except for errors'
    )
    
    return parser

if __name__ == "__main__":
    main()