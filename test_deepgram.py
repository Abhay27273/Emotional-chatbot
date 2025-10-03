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
from pipecat.services.deepgram.stt import DeepgramSTTService
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
logger.info("--- SMART TURN V3 + VOICE-AWARE VERSION (FRESH STT) ---")

app = FastAPI()

# --- LLM AND TTS SERVICES (MODULE LEVEL) ---
llm = OpenAILLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4.1",
    max_tokens=200,
    temperature=0.7,
)

tts = ElevenLabsTTSService(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice_id="O4cGUVdAocn0z4EpQ9yF",
    model="eleven_turbo_v2"
)

# --- TWILIO CREDENTIALS ---
try:
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    MY_NUMBER = os.getenv("TARGET_PHONE_NUMBER")
    BASE_URL = os.getenv("BASE_URL")
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER, MY_NUMBER]):
        raise ValueError("Missing required Twilio environment variables.")
    logger.info("Loaded Twilio credentials.")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    raise

# --- VOICE-AWARE SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are an empathetic AI assistant on a phone call. Your responses should adapt to the caller's voice characteristics:

VOICE ANALYSIS GUIDELINES:
- If the caller sounds energetic or speaks quickly: Match their energy with enthusiasm and concise responses
- If the caller sounds calm or soft-spoken: Use a gentle, thoughtful tone with measured pacing
- If the caller sounds frustrated or stressed: Be extra patient, validating, and solution-focused
- If the caller sounds cheerful or upbeat: Be warm and friendly, share their positive energy
- If the caller pauses frequently or speaks slowly: Give them space, don't rush, be patient

RESPONSE STYLE:
- Keep responses conversational and natural
- Mirror the caller's communication style (formal vs casual)
- Be concise but thorough
- Show genuine interest in their needs
- Use appropriate empathy based on their emotional tone

Your goal is to create a comfortable, human-like conversation that feels natural and responsive to the caller's voice and mood."""

# --- HELPER FUNCTIONS ---
def get_base_url(request: Request) -> str:
    """Determines base URL, prioritizing BASE_URL from environment."""
    if BASE_URL:
        logger.info(f"Using BASE_URL from environment: {BASE_URL}")
        return BASE_URL
    
    host = request.headers.get("host")
    protocol = request.headers.get("x-forwarded-proto", "http")
    if host:
        base_url = f"{protocol}://{host}"
        logger.info(f"Determined base URL from headers: {base_url}")
        return base_url
    
    logger.warning("Could not determine base URL")
    return "http://localhost:8000"

# --- FASTAPI ENDPOINTS ---

@app.post("/voice")
async def voice(request: Request):
    """Handles incoming calls from Twilio and connects them to WebSocket."""
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
        
        logger.info(f"*** RETURNING TwiML WITH WEBSOCKET URL: {full_ws_url} ***")
        return Response(content=twiml, media_type="application/xml")
    except Exception as e:
        logger.error(f"Error handling Twilio voice request: {e}", exc_info=True)
        return Response(status_code=500, content="Internal Server Error")


@app.get("/call")
async def make_call(request: Request):
    """Initiates an outbound call to the target phone number."""
    try:
        base_url = get_base_url(request)
        logger.info(f"=== INITIATING OUTBOUND CALL ===")
        logger.info(f"Base URL: {base_url}")

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            to=MY_NUMBER,
            from_=TWILIO_NUMBER,
            url=f"{base_url}/voice",
            status_callback=f"{base_url}/status",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST"
        )
        logger.info(f"✓ Call initiated successfully with SID: {call.sid}")
        return {"status": "calling", "sid": call.sid}
    except Exception as e:
        logger.error(f"✗ Error initiating outbound call: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/status")
async def status_callback(request: Request):
    """Logs call status from Twilio."""
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
        data = parse_qs(body_str)
        
        call_status = data.get("CallStatus", ["unknown"])[0]
        call_sid = data.get("CallSid", ["unknown"])[0]
        
        logger.info(f"Call {call_sid} status: {call_status}")
        
        if "ErrorCode" in data:
            error_code = data.get("ErrorCode", [""])[0]
            error_msg = data.get("ErrorMessage", [""])[0]
            logger.error(f"Call {call_sid} Error: {error_code} - {error_msg}")
            
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Error in status callback: {e}", exc_info=True)
        return Response(status_code=200)


@app.websocket("/ws/twilio")
async def twilio_ws(websocket: WebSocket):
    """Handles the WebSocket connection from Twilio with Smart Turn V3."""
    logger.info("=" * 80)
    logger.info("🔌 WEBSOCKET CONNECTION ATTEMPT RECEIVED")
    logger.info("=" * 80)
    
    await websocket.accept()
    logger.info("✓ WebSocket connection ACCEPTED from Twilio.")
    
    pipeline_task = None
    runner = PipelineRunner()

    try:
        # --- Handle initial Twilio messages ---
        logger.info("Waiting for 'connected' event...")
        connected_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        event = connected_msg.get("event")
        if event != "connected":
            logger.error(f"Expected 'connected' event, got '{event}'")
            await websocket.close(code=1003, reason="Expected 'connected' event")
            return
        logger.info(f"✓ Received 'connected' message")

        logger.info("Waiting for 'start' event...")
        start_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        event = start_msg.get("event")
        if event != "start":
            logger.error(f"Expected 'start' event, got '{event}'")
            await websocket.close(code=1003, reason="Expected 'start' event")
            return
        logger.info(f"✓ Received 'start' message")
        
        stream_sid = start_msg["start"]["streamSid"]
        call_sid = start_msg["start"]["callSid"]
        logger.info(f"Stream SID: {stream_sid}, Call SID: {call_sid}")

        # --- CREATE FRESH STT SERVICE FOR THIS CALL ---
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        logger.info(f"Creating fresh Deepgram STT service with key: {deepgram_key[:10]}...")
        
        stt = DeepgramSTTService(
            api_key=deepgram_key,
            model="nova-2-general",
            interim_results=True
        )
        logger.info("✓ Deepgram STT service created")

        # --- Configure Pipecat components ---
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

        # Configure Smart Turn V3 with optimized VAD
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        confidence=0.7,
                        start_secs=0.2,
                        stop_secs=0.5,
                        min_volume=0.6,
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

        # Pipeline with Smart Turn V3
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
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.warning("Client disconnected. Cancelling pipeline.")
            logger.info("Conversation history:")
            logger.info(context.get_messages_json())
            if pipeline_task:
                pipeline_task.cancel()

        @transport.event_handler("on_connection_error")
        async def on_connection_error(transport, error):
            logger.error(f"Connection error: {error}. Cancelling pipeline.")
            if pipeline_task:
                pipeline_task.cancel()

        logger.info("Starting pipeline with Smart Turn V3...")
        pipeline_task = asyncio.create_task(runner.run(task))

        # Wait for pipeline to initialize
        await asyncio.sleep(1.0)

        # Send initial greeting
        logger.info("Sending initial greeting...")
        await task.queue_frame(TextFrame("Hello! I'm your AI assistant. I can adapt to your communication style. How can I help you today?"))

        logger.info("Pipeline running with Smart Turn V3, waiting for completion...")
        await pipeline_task

    except asyncio.TimeoutError:
        logger.error("Timeout waiting for Twilio WebSocket messages")
        try:
            await websocket.close(code=1008, reason="Timeout")
        except:
            pass
    except asyncio.CancelledError:
        logger.warning("Pipeline task was cancelled.")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        from starlette.websockets import WebSocketState
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close(code=1011, reason="Internal Server Error")
            except:
                pass
    finally:
        logger.info("WebSocket connection handler finished.")
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()