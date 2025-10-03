import os
import asyncio
import logging
from dotenv import load_dotenv
from urllib.parse import parse_qs

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from twilio.rest import Client

from pipecat.frames.frames import TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner, PipelineTask
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

load_dotenv()

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipecat-app")
logger.info("--- LATENCY-OPTIMIZED WHISPER STT VERSION ---")

app = FastAPI()

# --- OPTIMIZED SYSTEM PROMPT (SHORTER FOR FASTER PROCESSING) ---
SYSTEM_PROMPT = """You are a helpful AI phone assistant. Be natural, concise, and conversational. Adapt your tone to match the caller. Keep responses brief and to the point."""

# --- TWILIO CREDENTIALS ---
try:
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    MY_NUMBER = os.getenv("TARGET_PHONE_NUMBER")
    BASE_URL = os.getenv("BASE_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER, MY_NUMBER]):
        raise ValueError("Missing required Twilio environment variables.")
    logger.info("Loaded credentials.")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    raise

# --- HELPER FUNCTIONS ---
def get_base_url(request: Request) -> str:
    if BASE_URL:
        return BASE_URL
    host = request.headers.get("host")
    protocol = request.headers.get("x-forwarded-proto", "http")
    return f"{protocol}://{host}" if host else "http://localhost:8000"

# --- FASTAPI ENDPOINTS ---

@app.post("/voice")
async def voice(request: Request):
    try:
        base_url = get_base_url(request)
        ws_protocol = "wss" if base_url.startswith("https") else "ws"
        ws_url_host = base_url.replace("https://", "").replace("http://", "")
        full_ws_url = f"{ws_protocol}://{ws_url_host}/ws/twilio"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Connect>
                <Stream url="{full_ws_url}"/>
            </Connect>
        </Response>"""
        
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error handling Twilio voice request: {e}", exc_info=True)
        return Response(status_code=500, content="Internal Server Error")


@app.get("/call")
async def make_call(request: Request):
    try:
        base_url = get_base_url(request)
        logger.info(f"Initiating call to {MY_NUMBER}")
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            to=MY_NUMBER,
            from_=TWILIO_NUMBER,
            url=f"{base_url}/voice",
            status_callback=f"{base_url}/status",
            status_callback_event=["initiated", "answered", "completed"],
            status_callback_method="POST"
        )
        logger.info(f"✓ Call SID: {call.sid}")
        return {"status": "calling", "sid": call.sid}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/status")
async def status_callback(request: Request):
    try:
        body = await request.body()
        data = parse_qs(body.decode('utf-8'))
        call_status = data.get("CallStatus", ["unknown"])[0]
        call_sid = data.get("CallSid", ["unknown"])[0]
        logger.info(f"Call {call_sid}: {call_status}")
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return Response(status_code=200)


@app.websocket("/ws/twilio")
async def twilio_ws(websocket: WebSocket):
    logger.info("WebSocket connection received")
    await websocket.accept()
    
    pipeline_task = None
    runner = PipelineRunner()

    try:
        # Fast event handling
        connected_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if connected_msg.get("event") != "connected":
            await websocket.close(code=1003)
            return
        
        start_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if start_msg.get("event") != "start":
            await websocket.close(code=1003)
            return
        
        stream_sid = start_msg["start"]["streamSid"]
        call_sid = start_msg["start"]["callSid"]
        logger.info(f"Call: {call_sid}")

        # --- OPTIMIZED SERVICES: Create per-call for clean state ---
        
        # Whisper STT - fastest model
        stt = OpenAISTTService(
            api_key=OPENAI_API_KEY,
            model="whisper-1"  # Already the fastest Whisper model
        )
        
        # LLM with aggressive optimization
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY,
            model="gpt-4o-mini",  # Fast model
            max_tokens=100,  # REDUCED from 200 - shorter responses = faster
            temperature=0.5,  # REDUCED from 0.7 - more deterministic = faster
        )
        
        # TTS with speed optimization
        tts = ElevenLabsTTSService(
            api_key=ELEVENLABS_API_KEY,
            voice_id="O4cGUVdAocn0z4EpQ9yF",
            model="eleven_turbo_v2",  # Already using turbo
            params=ElevenLabsTTSService.InputParams(
                optimize_streaming_latency=4,  # MAX optimization (0-4 scale)
                stability=0.5,  # REDUCED for speed
                similarity_boost=0.5,  # REDUCED for speed
            )
        )

        # Serializer
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            params=TwilioFrameSerializer.InputParams(
                sample_rate=8000,
                audio_codec="mulaw"
            ),
            call_sid=call_sid,
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN
        )

        # AGGRESSIVE VAD tuning for faster response
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        confidence=0.6,  # REDUCED - more aggressive detection
                        start_secs=0.1,  # REDUCED - detect speech faster
                        stop_secs=0.3,   # REDUCED - shorter pauses trigger end
                        min_volume=0.5,  # REDUCED - more sensitive
                    )
                ),
                turn_analyzer=LocalSmartTurnAnalyzerV3(),
                serializer=serializer,
            ),
        )

        context = OpenAILLMContext(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}]
        )
        
        context_aggregator = llm.create_context_aggregator(context)

        # Streamlined pipeline
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                audio_in_buffer_size=512,   # REDUCED buffer for faster processing
                audio_out_buffer_size=512,  # REDUCED buffer for faster output
                enable_metrics=False,  # DISABLED for slight performance gain
                enable_usage_metrics=False,  # DISABLED for slight performance gain
            ),
        )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Disconnected. History: {context.get_messages_json()}")
            if pipeline_task:
                pipeline_task.cancel()

        logger.info("Starting optimized pipeline...")
        pipeline_task = asyncio.create_task(runner.run(task))

        # Shorter initialization wait
        await asyncio.sleep(0.5)  # REDUCED from 1.0

        # Short greeting for speed
        await task.queue_frame(TextFrame("Hi! How can I help?"))

        await pipeline_task

    except asyncio.TimeoutError:
        logger.error("Timeout")
        try:
            await websocket.close(code=1008)
        except:
            pass
    except asyncio.CancelledError:
        logger.warning("Cancelled")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011)
        except:
            pass
    finally:
        logger.info("Connection finished")
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()