from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import torch
import torchaudio
import time
import os
import asyncio
import re
import statistics
import traceback 
from transformers import pipeline
from pipecat.frames.frames import Frame, EndFrame, AudioRawFrame, TextFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.base_transport import TransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global runner, transport, and connection for use in endpoints and startup
runner: PipelineRunner | None = None
transport: SmallWebRTCTransport | None = None
pipeline_task: PipelineTask | None = None
webrtc_connection: SmallWebRTCConnection | None = None

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager"""
    global runner, pipeline_task
    
    # Startup
    logger.info("🚀 Application starting up...")
    await startup_event()
    
    yield
    
    # Shutdown
    logger.info("🛑 Application shutting down...")
    if runner and pipeline_task:
        try:
            await runner.stop_when_done()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware ONCE - for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Client Interface ---
@app.get("/client")
async def client():
    """Serve the WebRTC client HTML interface"""
    try:
        # Try current directory first
        if os.path.exists("index.html"):
            logger.info("✅ Serving index.html from current directory")
            return FileResponse("index.html")
        
        # Try script directory
        file_path = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(file_path):
            logger.info(f"✅ Serving index.html from {file_path}")
            return FileResponse(file_path)
        
        # File not found
        logger.error("❌ index.html not found!")
        return JSONResponse(
            {"error": "index.html not found. Please create it in the same directory as this script."},
            status_code=404
        )
    except Exception as e:
        logger.error(f"Error serving client: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# --- SDP Offer/Answer exchange ---
@app.post("/offer")
async def offer(request: Request):
    """Handle WebRTC SDP offer from client"""
    global transport, webrtc_connection
    
    if transport is None or webrtc_connection is None:
        logger.error("❌ Transport or WebRTC connection not initialized")
        return JSONResponse({"error": "Transport not initialized"}, status_code=503)

    try:
        body = await request.json()
        logger.info("📡 Received SDP offer from client")
        logger.info(f"📦 Offer body keys: {body.keys()}")
        logger.info(f"📦 Offer type: {body.get('type')}")
        
        # Validate offer format
        if 'sdp' not in body or 'type' not in body:
            logger.error("❌ Invalid offer format - missing 'sdp' or 'type'")
            return JSONResponse({"error": "Invalid offer format"}, status_code=400)
        
        # Process the offer and generate answer using the internal method
        logger.info("🔄 Processing SDP offer...")
        
        # Create RTCSessionDescription from the offer
        from aiortc import RTCSessionDescription
        
        offer_desc = RTCSessionDescription(
            sdp=body["sdp"],
            type=body["type"]
        )
        
        # Use the webrtc_connection's peer connection to set remote description and create answer
        pc = webrtc_connection._pc  # Access the RTCPeerConnection
        
        await pc.setRemoteDescription(offer_desc)
        logger.info("✅ Remote description set")
        
        # Create answer
        answer_desc = await pc.createAnswer()
        await pc.setLocalDescription(answer_desc)
        logger.info("✅ Answer created and local description set")
        
        # Return the answer with ICE candidates
        answer = {
            "type": pc.localDescription.type,
            "sdp": pc.localDescription.sdp
        }
        
        logger.info(f"✅ SDP answer generated successfully")
        logger.info(f"📊 ICE gathering state: {pc.iceGatheringState}")
        logger.info(f"📊 ICE connection state: {pc.iceConnectionState}")
        
        return JSONResponse(answer)
    except Exception as e:
        logger.error(f"❌ Error processing offer: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        traceback.print_exc()
        return JSONResponse({"error": str(e), "error_type": type(e).__name__}, status_code=500)

# Root endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Pipecat WebRTC server is active",
        "endpoints": {
            "client": "/client (GET) - WebRTC UI",
            "offer": "/offer (POST) - WebRTC signaling",
            "health": "/health (GET) - Health check"
        }
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    global transport, runner, webrtc_connection
    return {
        "status": "healthy",
        "transport_initialized": transport is not None,
        "webrtc_connection_initialized": webrtc_connection is not None,
        "runner_initialized": runner is not None,
        "models": {
            "vad": model is not None,
            "whisper": WHISPER_PIPELINE is not None
        },
        "index_html_exists": os.path.exists("index.html")
    }

# --- Load Silero VAD ---
model = None
logger.info("Loading Silero VAD model...")
try:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    logger.info("✅ Silero VAD model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading Silero VAD model: {e}")
    logger.error("Server will continue but VAD features will be disabled")
    traceback.print_exc()

# --- Load Whisper Model ---
WHISPER_PIPELINE = None
logger.info("Loading Whisper model 'openai/whisper-small'...")
try:
    WHISPER_PIPELINE = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=0 if torch.cuda.is_available() else -1,
    )
    logger.info(f"✅ Whisper model loaded successfully on device: {WHISPER_PIPELINE.device}")
except Exception as e:
    logger.error(f"❌ Warning: Could not load Whisper pipeline. Error: {e}")
    traceback.print_exc()


def whisper_stt(audio_tensor, sr=16000, stream_mode=False):
    """
    Transcribes audio using Whisper.
    If stream_mode=True, returns partial transcript + word timings.
    Returns a dict: { 'text': str, 'chunks': [{start,end,word}, ...] }
    """
    if WHISPER_PIPELINE is None:
        return {"text": "error: whisper model not loaded", "chunks": []}
    
    audio_np = audio_tensor.squeeze().cpu().numpy()

    try:
        result = WHISPER_PIPELINE(
            {"array": audio_np, "sampling_rate": sr},
            chunk_length_s=15,
            stride_length_s=[4, 2],
            return_timestamps="word" if stream_mode else "sentence",
        )

        if isinstance(result, dict):
            text = result.get("text", "")
            chunks = result.get("chunks", [])
            return {"text": text, "chunks": chunks}
        else:
            return {"text": str(result), "chunks": []}

    except Exception as e:
        logger.error(f"Whisper STT failed: {e}")
        return {"text": "stt_error", "chunks": []}


def falling_pitch_enhanced(audio_tensor, sr=16000, min_frames=5, 
                          decline_threshold=0.85, slope_weight=0.3):
    """
    Enhanced pitch contour analysis for end-of-utterance detection.
    """
    try:
        pitch = torchaudio.functional.detect_pitch_frequency(
            audio_tensor, sample_rate=sr, frame_time=0.02
        )
        voiced_pitch = pitch[pitch > 0]
        if len(voiced_pitch) < min_frames: 
            return False
        
        if len(voiced_pitch) >= 3:
            voiced_pitch = torch.nn.functional.avg_pool1d(
                voiced_pitch.unsqueeze(0).unsqueeze(0), 
                kernel_size=3, stride=1, padding=1
            ).squeeze()
        
        first_segment = voiced_pitch[:max(2, len(voiced_pitch)//4)].mean()
        last_segment = voiced_pitch[-max(2, len(voiced_pitch)//4):].mean()
        ratio_declining = last_segment < first_segment * decline_threshold
        
        x = torch.arange(len(voiced_pitch), dtype=torch.float32)
        y = voiced_pitch
        n = len(x)
        if n > 1 and (x * x).sum() != x.sum() ** 2 / n:
            slope = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x * x).sum() - x.sum() ** 2)
            slope_declining = slope < -0.5
        else:
            slope_declining = False
        
        if len(voiced_pitch) > 1:
            diffs = voiced_pitch[1:] - voiced_pitch[:-1]
            declining_ratio = (diffs < 0).float().mean()
            contour_declining = declining_ratio > 0.6
        else:
            contour_declining = False
        
        confidence_score = (
            ratio_declining * 0.4 + 
            slope_declining * slope_weight + 
            contour_declining * (0.6 - slope_weight)
        )
        return confidence_score > 0.5
    except Exception as e:
        return False


class SmartTurnProcessor(FrameProcessor):
    """Processes audio frames to detect turn-taking using VAD, STT, and pitch analysis"""
    
    def __init__(self, sample_rate=16000, silence_threshold=0.7):
        super().__init__()
        self.sample_rate = sample_rate
        self.last_speech_time = None
        self.silence_threshold = silence_threshold
        self.in_speech = False
        self.audio_buffer = []
        self.silence_start_audio_time = None
        self.min_silence = 0.4
        self.max_silence = 1.2
        self.recent_word_durations = []

    def is_linguistically_complete(self, transcript: str) -> bool:
        """Check if transcript appears to be a complete utterance"""
        if not transcript or transcript.strip() == "":
            return False
        if transcript.strip().endswith((".", "?", "!")):
            return True
        
        incomplete_patterns = [
            r"\b(and|or|but|so|because|then|so on|then)\s*$",
            r"\b(in|on|at|by|for|with|about|to|from)\s*$",
            r"\b(that|which|who|whom|whose|where|when)\s*$",
            r"[,;:]$",
        ]
        for pat in incomplete_patterns:
            if re.search(pat, transcript, re.IGNORECASE):
                return False
        
        if len(transcript.split()) < 2:
            return False
        return True

    def lm_completion_score(self, transcript: str) -> float:
        """Simple language model completion score"""
        if not transcript:
            return 0.0
        if transcript.strip().endswith((".", "?", "!")):
            return 1.0
        return 0.5

    def _update_silence_threshold(self):
        """Dynamically adjust silence threshold based on speaking rate"""
        if not self.recent_word_durations:
            return
        median = statistics.median(self.recent_word_durations)
        mapped = median * 2.0
        mapped = max(self.min_silence, min(self.max_silence, mapped))
        self.silence_threshold = 0.8 * self.silence_threshold + 0.2 * mapped

    async def process_frame(self, frame: Frame, direction):
        """Process each frame for turn detection"""
        # Pass frame through the pipeline first
        await super().process_frame(frame, direction)

        # Only process audio frames
        if not isinstance(frame, AudioRawFrame):
            return

        # Check if VAD model is available
        if not model:
            return

        current_time = time.time()
        audio_time = getattr(frame, "timestamp", current_time)

        audio_chunk_1d = frame.audio.squeeze()
        self.audio_buffer.append(audio_chunk_1d)

        speech_detected = False
        try:
            speech_confidence = model(audio_chunk_1d, self.sample_rate).item()
            if speech_confidence > 0.5:
                speech_detected = True
        except Exception as e:
            # VAD inference failed, treat as silence
            pass

        if speech_detected:
            if not self.in_speech:
                logger.info("🎤 User started speaking")
            self.in_speech = True
            self.last_speech_time = current_time
            self.silence_start_audio_time = None
        else:  # silence detected
            if self.in_speech and self.last_speech_time:
                if self.silence_start_audio_time is None:
                    self.silence_start_audio_time = audio_time

                silence_dur = current_time - self.last_speech_time
                if silence_dur > self.silence_threshold:
                    try:
                        full_audio = torch.cat(self.audio_buffer, dim=0)
                        self.audio_buffer = []

                        stt_result = whisper_stt(full_audio, self.sample_rate, stream_mode=True)
                        transcript = stt_result.get("text", "").strip()

                        if transcript:
                            logger.info(f"📝 STT transcript: [{transcript}]")

                            durations = [
                                c["end"] - c["start"] 
                                for c in stt_result.get("chunks", []) 
                                if c.get("end") and c.get("start")
                            ]
                            if durations:
                                self.recent_word_durations.extend(durations)
                                self.recent_word_durations = self.recent_word_durations[-50:]
                                self._update_silence_threshold()

                            looks_complete = self.is_linguistically_complete(transcript)
                            lm_score = self.lm_completion_score(transcript)
                            pitch_fall = falling_pitch_enhanced(full_audio, self.sample_rate)
                            
                            completion_confidence = (
                                (0.7 if looks_complete else 0.0) + 
                                (0.3 if pitch_fall else 0.0)
                            )

                            if completion_confidence > 0.7:
                                logger.info(f"✅ User finished speaking (turn end): {transcript}")
                                
                                # Push transcript as TextFrame
                                await self.push_frame(TextFrame(text=transcript))
                                
                                self.in_speech = False
                                self.silence_start_audio_time = None
                            else:
                                logger.info(f"⏸️  Possible pause, transcript so far: {transcript}")
                                # Re-add audio for continued listening
                                self.audio_buffer.append(full_audio)
                        else:
                            # Empty transcription, reset
                            self.in_speech = False
                            self.audio_buffer = []

                    except Exception as e:
                        logger.error(f"Audio processing failed: {e}")
                        traceback.print_exc()
                        self.audio_buffer = []
                        self.in_speech = False
                        self.silence_start_audio_time = None


async def startup_event():
    """
    Sets up and runs the PipeCat pipeline using WebRTC transport in a background task.
    """
    global runner, transport, pipeline_task, webrtc_connection
    
    # Model check
    if not model or not WHISPER_PIPELINE:
        logger.warning("⚠️  Models failed to load. Core logic (VAD/STT) may fail.")

    # Audio configuration
    RATE = 16000
    
    logger.info("🔧 Initializing pipeline components...")
    
    # 1. Create processor
    std_processor = SmartTurnProcessor(sample_rate=RATE, silence_threshold=0.7)

    # 2. Setup WebRTC connection and transport with enhanced ICE servers
    transport_params = TransportParams(
        audio_in_enabled=True, 
        audio_out_enabled=True,
        audio_in_sample_rate=RATE,
        audio_out_sample_rate=RATE,
    )
    
    # Enhanced ICE server configuration with STUN and TURN servers
    # Use simple string format for STUN servers and dict format for TURN servers
    ice_servers = [
        # Google STUN servers (simple string format)
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
    ]
    
    logger.info(f"🌐 Configuring ICE servers: {len(ice_servers)} STUN servers")
    logger.info("Note: TURN servers configured in client only (server-side uses STUN)")
    
    # Create and store the webrtc_connection globally with enhanced ICE configuration
    webrtc_connection = SmallWebRTCConnection(
        ice_servers=ice_servers
    )
    
    # Initialize transport with the connection
    transport = SmallWebRTCTransport(
        params=transport_params, 
        webrtc_connection=webrtc_connection,
    )
    
    logger.info("✅ WebRTC transport initialized with enhanced ICE configuration")
    logger.info("✅ WebRTC connection initialized")
    
    # 3. Create pipeline
    pipeline = Pipeline(
        processors=[
            transport.input(),
            std_processor,
            transport.output(),
        ]
    )
    
    logger.info("✅ Pipeline created")
    
    # 4. Create task and runner
    pipeline_task = PipelineTask(pipeline=pipeline)
    runner = PipelineRunner(handle_sigint=False)
    
    # 5. Start the pipeline in a background task
    logger.info("🚀 Starting WebRTC PipeCat Pipeline...")
    asyncio.create_task(runner.run(pipeline_task))
    logger.info("✅ Pipeline running in background")


# --- Run FastAPI server if executed directly ---
if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 60)
    logger.info("🌐 Starting FastAPI server on http://0.0.0.0:8000")
    logger.info("🎨 WebRTC client UI: http://localhost:8000/client")
    logger.info("📡 WebRTC signaling: http://localhost:8000/offer")
    logger.info("💚 Health check: http://localhost:8000/health")
    logger.info("=" * 60)
    logger.info("")
    logger.info("🔍 Troubleshooting tips:")
    logger.info("   - If running in Codespaces, use the forwarded URL instead of localhost")
    logger.info("   - Check that port 8000 is properly forwarded")
    logger.info("   - For persistent connection issues, consider setting up a TURN server")
    logger.info("=" * 60)
    uvicorn.run("test_st:app", host="0.0.0.0", port=8000, reload=False)