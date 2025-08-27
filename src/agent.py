import logging
import os

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    ChatContext
)
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
    elevenlabs,
    azure
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.utils.audio import audio_frames_from_file

from config import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    OPENAI_MODEL,
    LOG_LEVEL,
)

logger = logging.getLogger("voice-agent")
logger.setLevel(LOG_LEVEL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DISCLAIMER_PATH = os.path.join(BASE_DIR, "assets", "disclaimer.wav")

from db import get_customer_by_phone

class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        
        # Azure STT (multilingual).
        stt = azure.STT(
            speech_key=AZURE_SPEECH_KEY,
            speech_region=AZURE_SPEECH_REGION,
            # languages=["en-US", "ar-KW"],  # hints (STT still auto-detects best effort)
            # detect_language is implied by multiple languages in new plugins.
        )

        # ElevenLabs TTS (multilingual).
        tts = elevenlabs.TTS(
            api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            model="eleven_turbo_v2_5",
            streaming_latency=2,
            # Optional: adjust chunk schedule for latency vs quality
            chunk_length_schedule=[80, 120, 200, 260],  
        )

        super().__init__(
            instructions="You are **Khalid (خالد)**, a twenty six year old front desk agent for a restaurant in Kuwait."
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. ",
            chat_ctx=chat_ctx,
            stt=stt,
            llm=openai.LLM(model=OPENAI_MODEL),
            tts=tts,
            vad=silero.VAD.load(),
            # use LiveKit's transformer-based turn detector
            turn_detection=MultilingualModel(),
        )

    async def on_enter(self):
        customer_data = await get_customer_by_phone("+923360048001")

        audio_iterable = audio_frames_from_file(DISCLAIMER_PATH)
        
        # Disclaimer audio 
        handle = await self.session.say(
            "",
            audio=audio_iterable,
            allow_interruptions=False
        )
        await handle.wait_for_playout()

        # The agent should be polite and greet the user when it joins :)
        greeting = "السلام عليكم وحياك الله في مطعمنا، معاك خالد من الاستقبال. شلون أقدر أخدمك؟"
        self.session.generate_reply(
            instructions="Hey, how can I help you today?", allow_interruptions=True
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # enable background voice & noise cancellation, powered by Krisp
            # included at no additional cost with LiveKit Cloud
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="inbound-agent",
        ),
    )