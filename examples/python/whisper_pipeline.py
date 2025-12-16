"""
Whisper-based Voice Command Pipeline for Humanoid Robots

This script implements a complete voice command processing pipeline using OpenAI's Whisper
model for humanoid robot applications. It includes audio capture, speech recognition,
command parsing, and integration with robot control systems.

The pipeline handles:
- Real-time audio capture and processing
- Speech-to-text conversion using Whisper
- Natural language command parsing
- Robot command execution
- Error handling and feedback
"""

import asyncio
import pyaudio
import numpy as np
import whisper
import torch
import queue
import threading
import time
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration parameters"""
    rate: int = 16000  # Sampling rate
    chunk: int = 1024  # Frames per buffer
    channels: int = 1  # Mono audio
    format: int = pyaudio.paInt16  # 16-bit format
    input_device_index: Optional[int] = None  # Input device index

@dataclass
class WhisperConfig:
    """Whisper model configuration"""
    model_size: str = "base"  # Model size: tiny, base, small, medium, large
    device: str = None  # Device: cpu, cuda, auto
    language: str = "en"  # Language code
    beam_size: int = 5  # Beam size for decoding
    temperature: float = 0.0  # Temperature for sampling

@dataclass
class RobotCommand:
    """Represents a parsed robot command"""
    action: str
    parameters: Dict[str, Any]
    confidence: float
    original_text: str
    timestamp: float

class AudioCapture:
    """Handles audio capture from microphone"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.listen_for_speech = True

        # Initialize audio stream
        self._initialize_stream()

        # Audio analysis parameters
        self.silence_threshold = 500  # Adjust based on microphone sensitivity
        self.min_audio_length = 8000  # Minimum 0.5 seconds at 16kHz
        self.max_audio_length = 48000  # Maximum 3 seconds at 16kHz

    def _initialize_stream(self):
        """Initialize the audio stream"""
        try:
            self.stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.chunk,
                input_device_index=self.config.input_device_index
            )
            logger.info(f"Audio stream initialized at {self.config.rate}Hz")
        except Exception as e:
            logger.error(f"Error initializing audio stream: {e}")
            raise

    def start_recording(self):
        """Start audio recording in a separate thread"""
        self.recording = True
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()
        logger.info("Audio recording started")

    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        logger.info("Audio recording stopped")

    def _record_audio(self):
        """Internal method to record audio in a loop"""
        silence_frames = 0
        silence_threshold_frames = int(1.0 * self.config.rate / self.config.chunk)  # 1 second of silence

        while self.recording:
            try:
                data = self.stream.read(self.config.chunk, exception_on_overflow=False)
                self.audio_queue.put(data)

                # Check for silence to optimize processing
                audio_data = np.frombuffer(data, dtype=np.int16)
                max_amplitude = np.max(np.abs(audio_data))

                if max_amplitude < self.silence_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0  # Reset silence counter when speech detected

                # If we have a significant amount of silence, we might want to pause processing
                if silence_frames > silence_threshold_frames and self.listen_for_speech:
                    # This could be used to implement a "sleep" mode to save resources
                    pass

            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                break

    def get_audio_data(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get accumulated audio data from the queue"""
        frames = []
        try:
            while True:
                data = self.audio_queue.get(timeout=timeout)
                frames.append(data)
        except queue.Empty:
            pass

        if frames:
            audio_data = b''.join(frames)
            return np.frombuffer(audio_data, dtype=np.int16)
        return None

    def close(self):
        """Close the audio stream and terminate PyAudio"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        logger.info("Audio resources closed")

class WhisperProcessor:
    """Handles speech recognition using Whisper model"""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Whisper model '{config.model_size}' on {self.device}...")
        self.model = whisper.load_model(config.model_size, device=self.device)
        logger.info("Whisper model loaded successfully")

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[Dict]:
        """
        Transcribe audio data using Whisper

        Args:
            audio_data: Audio samples as numpy array (assumed to be at 16kHz)

        Returns:
            Dictionary containing transcription results or None if failed
        """
        try:
            # Ensure audio is in the right format
            if audio_data.dtype != np.float32:
                # Convert to float32 and normalize
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Run transcription
            result = self.model.transcribe(
                audio_data,
                language=self.config.language,
                beam_size=self.config.beam_size,
                temperature=self.config.temperature,
                fp16=(self.device == "cuda")
            )

            return result
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            return None

    def transcribe_file(self, audio_file_path: str) -> Optional[Dict]:
        """Transcribe an audio file"""
        try:
            result = self.model.transcribe(audio_file_path, language=self.config.language)
            return result
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return None

class CommandParser:
    """Parses natural language commands into robot actions"""

    def __init__(self):
        # Define command patterns and their corresponding actions
        self.command_patterns = {
            # Navigation commands
            'move_to': [
                r'move to (.+)',
                r'go to (.+)',
                r'go over to (.+)',
                r'walk to (.+)',
                r'navigate to (.+)',
                r'go to the (.+)',
                r'travel to (.+)'
            ],

            # Movement commands
            'move_forward': [
                r'move forward',
                r'go forward',
                r'go straight',
                r'walk forward',
                r'move ahead',
                r'go ahead',
                r'forward'
            ],

            'move_backward': [
                r'move backward',
                r'go backward',
                r'go back',
                r'back up',
                r'move back',
                r'backward'
            ],

            'turn_left': [
                r'turn left',
                r'turn to the left',
                r'rotate left',
                r'pivot left',
                r'left'
            ],

            'turn_right': [
                r'turn right',
                r'turn to the right',
                r'rotate right',
                r'pivot right',
                r'right'
            ],

            # Manipulation commands
            'pick_up': [
                r'pick up (.+)',
                r'grab (.+)',
                r'take (.+)',
                r'get (.+)',
                r'pick (.+) up',
                r'lift (.+)'
            ],

            'put_down': [
                r'put down (.+)',
                r'drop (.+)',
                r'release (.+)',
                r'put (.+) down',
                r'place (.+)'
            ],

            # Social commands
            'wave': [
                r'wave',
                r'wave hello',
                r'say hello',
                r'hello',
                r'greet'
            ],

            'dance': [
                r'dance',
                r'do a dance',
                r'move to music',
                r'perform'
            ],

            # Control commands
            'stop': [
                r'stop',
                r'freeze',
                r'hold',
                r'wait',
                r'pause',
                r'break'
            ],

            'follow': [
                r'follow me',
                r'follow (.+)',
                r'come with me',
                r'follow behind'
            ],

            # Information commands
            'battery_level': [
                r'battery level',
                r'how much battery',
                r'battery status',
                r'power level'
            ],

            'time': [
                r'what time is it',
                r'current time',
                r'tell me the time',
                r'what time'
            ]
        }

        # Location mappings
        self.location_mappings = {
            'kitchen': 'kitchen_location',
            'living room': 'living_room_location',
            'bedroom': 'bedroom_location',
            'office': 'office_location',
            'bathroom': 'bathroom_location',
            'dining room': 'dining_room_location',
            'hallway': 'hallway_location',
            'entrance': 'entrance_location',
            'exit': 'exit_location',
            'table': 'table_location',
            'counter': 'counter_location',
            'couch': 'couch_location',
            'chair': 'chair_location'
        }

        # Object mappings
        self.object_mappings = {
            'bottle': 'bottle_object',
            'cup': 'cup_object',
            'book': 'book_object',
            'phone': 'phone_object',
            'keys': 'keys_object',
            'toy': 'toy_object',
            'food': 'food_object',
            'water': 'water_object',
            'apple': 'apple_object',
            'banana': 'banana_object',
            'milk': 'milk_object',
            'juice': 'juice_object'
        }

    def parse_command(self, text: str) -> Optional[RobotCommand]:
        """
        Parse a natural language command and return a RobotCommand object

        Args:
            text: The natural language command text

        Returns:
            RobotCommand object or None if no command is recognized
        """
        if not text:
            return None

        text = text.lower().strip()
        original_text = text

        # Try each action type
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    # Extract parameters if any
                    parameters = self._extract_parameters(action, match, text)

                    # Calculate confidence based on pattern match quality
                    confidence = self._calculate_confidence(action, text, match)

                    return RobotCommand(
                        action=action,
                        parameters=parameters,
                        confidence=confidence,
                        original_text=original_text,
                        timestamp=time.time()
                    )

        return None

    def _extract_parameters(self, action: str, match: re.Match, original_text: str) -> Dict[str, Any]:
        """Extract parameters from the matched command"""
        parameters = {}

        if action in ['move_to', 'pick_up', 'put_down', 'follow']:
            # Extract object/location name
            if match.groups():
                param_text = match.group(1).strip()

                # Try to map to known locations/objects
                if action == 'move_to':
                    # Check if it's a known location
                    for known_loc, mapped_loc in self.location_mappings.items():
                        if known_loc in param_text:
                            parameters['location'] = mapped_loc
                            break
                    else:
                        # Use the raw location name
                        parameters['location'] = param_text

                elif action in ['pick_up', 'put_down']:
                    # Check if it's a known object
                    for known_obj, mapped_obj in self.object_mappings.items():
                        if known_obj in param_text:
                            parameters['object'] = mapped_obj
                            break
                    else:
                        # Use the raw object name
                        parameters['object'] = param_text

                elif action == 'follow':
                    parameters['target'] = param_text

        # Add additional context parameters
        parameters['timestamp'] = time.time()
        parameters['original_text'] = original_text

        return parameters

    def _calculate_confidence(self, action: str, text: str, match: re.Match) -> float:
        """Calculate confidence score for the command match"""
        # Base confidence on match length and text length
        match_length = len(match.group(0))
        text_length = len(text)

        # Calculate base confidence
        base_confidence = min(1.0, match_length / text_length)

        # Adjust based on action type
        action_weights = {
            'move_to': 0.9,
            'move_forward': 0.8,
            'move_backward': 0.8,
            'turn_left': 0.8,
            'turn_right': 0.8,
            'pick_up': 0.9,
            'put_down': 0.9,
            'wave': 0.7,
            'stop': 0.9,
            'follow': 0.9,
            'dance': 0.7,
            'battery_level': 0.8,
            'time': 0.8
        }

        weight = action_weights.get(action, 0.7)
        confidence = base_confidence * weight

        # Ensure minimum confidence
        return max(0.3, confidence)

    def get_suggested_commands(self) -> List[str]:
        """Get a list of suggested commands for users"""
        suggestions = [
            "Move to the kitchen",
            "Go to the living room",
            "Pick up the bottle",
            "Turn left",
            "Move forward",
            "Stop",
            "Follow me",
            "Wave hello",
            "What time is it?",
            "Battery level"
        ]
        return suggestions

class RobotController:
    """Simulates robot control - in real implementation, this would interface with actual robot"""

    def __init__(self):
        self.current_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.battery_level = 100.0
        self.is_moving = False
        self.is_executing_command = False

    async def execute_command(self, command: RobotCommand) -> bool:
        """Execute a parsed robot command"""
        action = command.action
        params = command.parameters

        logger.info(f"Executing command: {action} with params: {params}")

        try:
            if action == 'move_to':
                return await self._execute_move_to(params)
            elif action == 'move_forward':
                return await self._execute_move_forward(params)
            elif action == 'move_backward':
                return await self._execute_move_backward(params)
            elif action == 'turn_left':
                return await self._execute_turn_left(params)
            elif action == 'turn_right':
                return await self._execute_turn_right(params)
            elif action == 'pick_up':
                return await self._execute_pick_up(params)
            elif action == 'put_down':
                return await self._execute_put_down(params)
            elif action == 'wave':
                return await self._execute_wave(params)
            elif action == 'dance':
                return await self._execute_dance(params)
            elif action == 'stop':
                return await self._execute_stop(params)
            elif action == 'follow':
                return await self._execute_follow(params)
            elif action == 'battery_level':
                return await self._execute_battery_level(params)
            elif action == 'time':
                return await self._execute_time(params)
            else:
                logger.warning(f'Unknown command action: {action}')
                return False
        except Exception as e:
            logger.error(f'Error executing command {action}: {e}')
            return False

    async def _execute_move_to(self, params):
        """Execute move to location command"""
        location = params.get('location', 'unknown')
        logger.info(f'Moving to {location}')

        # Simulate movement time
        await asyncio.sleep(2)

        # Update position (in real system, this would be from actual sensors)
        if 'kitchen' in location:
            self.current_position = {"x": 2.0, "y": 1.0, "z": 0.0}
        elif 'living' in location:
            self.current_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        elif 'bedroom' in location:
            self.current_position = {"x": -2.0, "y": 1.0, "z": 0.0}

        return True

    async def _execute_move_forward(self, params):
        """Execute move forward command"""
        logger.info('Moving forward')
        await asyncio.sleep(1)
        # Update position
        self.current_position["y"] += 0.5
        return True

    async def _execute_move_backward(self, params):
        """Execute move backward command"""
        logger.info('Moving backward')
        await asyncio.sleep(1)
        # Update position
        self.current_position["y"] -= 0.5
        return True

    async def _execute_turn_left(self, params):
        """Execute turn left command"""
        logger.info('Turning left')
        await asyncio.sleep(0.5)
        return True

    async def _execute_turn_right(self, params):
        """Execute turn right command"""
        logger.info('Turning right')
        await asyncio.sleep(0.5)
        return True

    async def _execute_pick_up(self, params):
        """Execute pick up object command"""
        obj = params.get('object', 'unknown')
        logger.info(f'Picking up {obj}')
        await asyncio.sleep(2)
        return True

    async def _execute_put_down(self, params):
        """Execute put down object command"""
        obj = params.get('object', 'unknown')
        logger.info(f'Putting down {obj}')
        await asyncio.sleep(2)
        return True

    async def _execute_wave(self, params):
        """Execute wave command"""
        logger.info('Waving')
        await asyncio.sleep(1)
        return True

    async def _execute_dance(self, params):
        """Execute dance command"""
        logger.info('Dancing')
        await asyncio.sleep(3)
        return True

    async def _execute_stop(self, params):
        """Execute stop command"""
        logger.info('Stopping')
        self.is_moving = False
        return True

    async def _execute_follow(self, params):
        """Execute follow command"""
        target = params.get('target', 'unknown')
        logger.info(f'Following {target}')
        self.is_moving = True
        await asyncio.sleep(2)
        return True

    async def _execute_battery_level(self, params):
        """Execute battery level command"""
        logger.info(f'Battery level: {self.battery_level}%')
        # In a real system, this would query the actual battery
        return True

    async def _execute_time(self, params):
        """Execute time command"""
        current_time = time.strftime("%H:%M")
        logger.info(f'Current time: {current_time}')
        return True

class VoiceCommandPipeline:
    """Main pipeline that orchestrates the entire voice command processing"""

    def __init__(self,
                 audio_config: AudioConfig = None,
                 whisper_config: WhisperConfig = None):
        self.audio_config = audio_config or AudioConfig()
        self.whisper_config = whisper_config or WhisperConfig()

        # Initialize components
        self.audio_capture = AudioCapture(self.audio_config)
        self.whisper_processor = WhisperProcessor(self.whisper_config)
        self.command_parser = CommandParser()
        self.robot_controller = RobotController()

        # Processing state
        self.running = False
        self.processing_thread = None
        self.audio_buffer = np.array([])

        # Callbacks
        self.command_callbacks = []
        self.error_callbacks = []

        # Performance metrics
        self.metrics = {
            'audio_samples_processed': 0,
            'commands_processed': 0,
            'transcription_time_avg': 0.0,
            'command_execution_time_avg': 0.0
        }

    def add_command_callback(self, callback: Callable[[RobotCommand], None]):
        """Add a callback function for processed commands"""
        self.command_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str], None]):
        """Add a callback function for errors"""
        self.error_callbacks.append(callback)

    def start(self):
        """Start the voice command pipeline"""
        self.running = True
        self.audio_capture.start_recording()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Voice command pipeline started")

    def stop(self):
        """Stop the voice command pipeline"""
        self.running = False
        self.audio_capture.stop_recording()
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Voice command pipeline stopped")

    def _processing_loop(self):
        """Main processing loop running in a separate thread"""
        while self.running:
            try:
                # Get audio data from capture
                audio_data = self.audio_capture.get_audio_data(timeout=0.01)

                if audio_data is not None and len(audio_data) > 0:
                    # Add to buffer for continuous processing
                    self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

                    # Check if we have enough audio for processing
                    if len(self.audio_buffer) >= self.audio_capture.min_audio_length:
                        # Check for speech activity to trigger processing
                        if self._should_process_audio():
                            self._process_audio_buffer()

                    # Limit buffer size to prevent excessive memory usage
                    if len(self.audio_buffer) > self.audio_capture.max_audio_length:
                        # Keep only the most recent audio
                        self.audio_buffer = self.audio_buffer[-self.audio_capture.max_audio_length:]

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self._trigger_error_callbacks(f"Processing loop error: {e}")

            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def _should_process_audio(self) -> bool:
        """Determine if we should process the current audio buffer"""
        # Check if there's sufficient speech activity
        if len(self.audio_buffer) < self.audio_capture.min_audio_length:
            return False

        # Calculate average amplitude to detect speech
        avg_amplitude = np.mean(np.abs(self.audio_buffer))
        return avg_amplitude > self.audio_capture.silence_threshold * 0.5

    def _process_audio_buffer(self):
        """Process the current audio buffer"""
        start_time = time.time()

        try:
            # Transcribe the audio
            result = self.whisper_processor.transcribe_audio(self.audio_buffer)

            if result and result.get('text', '').strip():
                transcription = result['text'].strip()
                logger.info(f"Transcribed: {transcription}")

                # Parse the command
                robot_command = self.command_parser.parse_command(transcription)

                if robot_command and robot_command.confidence > 0.5:
                    logger.info(f"Parsed command: {robot_command.action} (confidence: {robot_command.confidence:.2f})")

                    # Execute the command asynchronously
                    asyncio.run(self._execute_command_async(robot_command))

                    # Trigger command callbacks
                    self._trigger_command_callbacks(robot_command)

                    # Update metrics
                    execution_time = time.time() - start_time
                    self._update_command_execution_metrics(execution_time)
                else:
                    logger.info(f"Command not recognized or low confidence: {transcription}")
            else:
                logger.info("No speech detected or transcription failed")

        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
            self._trigger_error_callbacks(f"Audio processing error: {e}")
        finally:
            # Clear the buffer for next round
            self.audio_buffer = np.array([])

    async def _execute_command_async(self, command: RobotCommand):
        """Execute command asynchronously"""
        start_time = time.time()
        success = await self.robot_controller.execute_command(command)
        execution_time = time.time() - start_time

        if success:
            logger.info(f"Command executed successfully: {command.action}")
        else:
            logger.error(f"Command execution failed: {command.action}")
            self._trigger_error_callbacks(f"Command execution failed: {command.action}")

    def _trigger_command_callbacks(self, command: RobotCommand):
        """Trigger all command callbacks"""
        for callback in self.command_callbacks:
            try:
                callback(command)
            except Exception as e:
                logger.error(f"Error in command callback: {e}")

    def _trigger_error_callbacks(self, error_msg: str):
        """Trigger all error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error_msg)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def _update_command_execution_metrics(self, execution_time: float):
        """Update command execution metrics"""
        self.metrics['commands_processed'] += 1
        old_avg = self.metrics['command_execution_time_avg']
        new_avg = ((old_avg * (self.metrics['commands_processed'] - 1)) + execution_time) / self.metrics['commands_processed']
        self.metrics['command_execution_time_avg'] = new_avg

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()

    def get_suggested_commands(self) -> List[str]:
        """Get suggested commands for users"""
        return self.command_parser.get_suggested_commands()

def main():
    """Main function to run the Whisper voice command pipeline"""
    logger.info("Starting Whisper Voice Command Pipeline for Humanoid Robots")

    # Create configuration
    audio_config = AudioConfig(
        rate=16000,
        chunk=1024,
        channels=1
    )

    whisper_config = WhisperConfig(
        model_size="base",  # Use "base" for good balance of speed/accuracy
        device=None,  # Auto-detect GPU if available
        language="en"
    )

    # Create the pipeline
    pipeline = VoiceCommandPipeline(audio_config, whisper_config)

    # Add callbacks
    def on_command(command: RobotCommand):
        print(f"Executed command: {command.action} - {command.original_text}")

    def on_error(error: str):
        print(f"Error occurred: {error}")

    pipeline.add_command_callback(on_command)
    pipeline.add_error_callback(on_error)

    # Print suggested commands
    print("\nSuggested commands:")
    for cmd in pipeline.get_suggested_commands():
        print(f"  - {cmd}")
    print("\nSpeak one of these commands, or say 'quit' to exit.\n")

    try:
        # Start the pipeline
        pipeline.start()

        # Keep running until user interrupts
        print("Pipeline is running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)

            # Check for quit command (this would normally come from voice)
            # For demo purposes, we'll check for a quit file
            import os
            if os.path.exists("quit_pipeline"):
                print("Quit signal received")
                break

    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.stop()
        print("Pipeline stopped.")

        # Print final metrics
        metrics = pipeline.get_metrics()
        print(f"\nFinal Metrics:")
        print(f"  Commands processed: {metrics['commands_processed']}")
        print(f"  Avg execution time: {metrics['command_execution_time_avg']:.2f}s")

if __name__ == "__main__":
    main()