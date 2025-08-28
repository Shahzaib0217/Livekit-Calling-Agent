import logging
import os

from typing import Optional, Any
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
    ChatContext,
    function_tool,
    RunContext
)
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
    elevenlabs,
    azure
)
from livekit.agents.utils.audio import audio_frames_from_file
from livekit import api
from livekit.protocol import sip

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

from db import get_customer_by_phone, create_cart, update_cart

class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext, participant_identity: str, room_name: str, cart: dict) -> None:
        self.participant_identity = participant_identity
        self.room_name = room_name
        self.cart = cart
        
        # Azure STT (multilingual).
        stt = azure.STT(
            speech_key=AZURE_SPEECH_KEY,
            speech_region=AZURE_SPEECH_REGION,
            # languages=["en-US", "ar-KW"],
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
            instructions="""You are **Khalid (خالد)**, a polite and concise front desk agent for a restaurant in Kuwait.
            Speak naturally, keep answers short, and avoid unnecessary punctuation.
            
            You can help customers with:
            - Taking orders using the **add_to_cart** tool
            - Showing cart contents using the **view_cart** tool
            - Answering questions about menu items
            
            CRITICAL: When you call a function tool, IMMEDIATELY speak the exact result returned by the function.
            Do not add extra commentary or wait. Just say exactly what the function returned.
            
            Use the **transfer_call** tool ONLY when the user explicitly asks to talk to a manager,
            speak to a real person, or requests escalation.
            Do NOT offer to transfer unless the customer insists.
            
            When taking orders, always confirm the item name and quantity before adding to cart.""",
            chat_ctx=chat_ctx,
            stt=stt,
            llm=openai.LLM(model=OPENAI_MODEL),
            tts=tts,
            vad=silero.VAD.load(),
            # use LiveKit's transformer-based turn detector
            # turn_detection=MultilingualModel(),
        )

    async def on_enter(self):
        # TODO: Get actual phone number from SIP context or participant metadata
        customer_data = await get_customer_by_phone("+923360048001")
        
        # Associate customer with cart if found
        if customer_data:
            self.cart["customer_id"] = customer_data["id"]
            logger.info(f"Associated cart with customer: {customer_data.get('first_name', '')} {customer_data.get('last_name', '')}")

        audio_iterable = audio_frames_from_file(DISCLAIMER_PATH)
        
        # Disclaimer audio 
        # handle = await self.session.say(
        #     "",
        #     audio=audio_iterable,
        #     allow_interruptions=False
        # )
        # await handle.wait_for_playout()

        # The agent should be polite and greet the user when it joins :)
        greeting = "السلام عليكم وحياك الله في مطعمنا، معاك خالد من الاستقبال. شلون أقدر أخدمك؟"
        self.session.say(
            greeting, allow_interruptions=True
        )

    @function_tool()
    async def add_to_cart(
        self,
        context: RunContext,
        product_id: str,
        quantity: int = 1,
        special_instructions: str = ""
    ) -> str:
        """
        Add an item to the customer's cart or update quantity if item already exists.
        
        Args:
            product_id: The ID of the product to add
            quantity: Number of items to add (must be positive)
            special_instructions: Any special instructions for the item
            
        Returns:
            Confirmation message about the cart update
        """
        try:
            # Validate inputs
            if not product_id or not isinstance(product_id, str):
                return "Invalid product ID provided."
            
            if quantity <= 0:
                return "Quantity must be a positive number."
            
            logger.info(f"Adding to cart: product_id={product_id}, quantity={quantity}")
            
            self.cart = await update_cart(
                cart=self.cart,
                product_id=product_id,
                quantity=quantity,
                selected_options=[],  # For now, we'll handle options separately
                special_instructions=special_instructions or ""
            )
            
            # Find the updated item to provide feedback
            updated_item = next((item for item in self.cart["items"] if item["product_id"] == product_id), None)
            
            if updated_item:
                return f"Added {updated_item['name']} (quantity: {updated_item['quantity']}) to your cart. Cart total: {self.cart['total_amount']} KWD"
            else:
                return "Item added to cart successfully"
                
        except Exception as e:
            logger.error(f"Error adding to cart: {e}")
            return f"Sorry, I couldn't add that item to your cart. {str(e)}"

    @function_tool()
    async def view_cart(self, context: RunContext) -> str:
        """
        View the current contents of the customer's cart.
        
        Returns:
            str: Summary of cart contents and total
        """
        try:
            logger.info(f"Viewing cart with {len(self.cart['items'])} items")
            
            if not self.cart["items"]:
                result = "Your cart is empty right now."
                logger.info(f"Cart view result: {result}")
                return result
            
            if len(self.cart["items"]) == 1:
                item = self.cart["items"][0]
                cart_summary = f"You have {item['quantity']} {item['name']} in your cart for {self.cart['total_amount']} KWD."
            else:
                cart_summary = f"You have {len(self.cart['items'])} items in your cart for a total of {self.cart['total_amount']} KWD."
            
            logger.info(f"Cart view result: {cart_summary}")
            return cart_summary
            
        except Exception as e:
            logger.error(f"Error viewing cart: {e}")
            return "Sorry, I couldn't retrieve your cart information."

    @function_tool()
    async def transfer_call(self, context: RunContext) -> None:
        """
        Transfer the SIP call to another number.

        Args:
            participant_identity (str): The identity of the participant.
            transfer_to (str): The phone number to transfer the call to.
        """

        logger.info(f"Transferring call for participant {self.participant_identity}")

        try:
            livekit_url = os.getenv('LIVEKIT_URL')
            api_key = os.getenv('LIVEKIT_API_KEY')
            api_secret = os.getenv('LIVEKIT_API_SECRET')
            livekit_api = api.LiveKitAPI(
                url=livekit_url,
                api_key=api_key,
                api_secret=api_secret
            )

            transfer_request = sip.TransferSIPParticipantRequest(
                participant_identity=self.participant_identity,
                room_name=self.room_name,
                # we will dynamically fetch transfer_to from the branches phone number. 
                # transfer_to="tel:+14246675385",
                play_dialtone=True
            )

            await livekit_api.sip.transfer_sip_participant(transfer_request)

        except Exception as e:
            logger.error(f"Failed to transfer call: {e}", exc_info=True)
            await self.session.generate_reply(user_input="I'm sorry, I couldn't transfer your call. Is there something else I can help with?")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # TODO: Replace with actual location_id from context or configuration
    location_id = "22222222-2222-2222-2222-222222222222"
    cart = await create_cart(location_id=location_id)
    logger.info(f"Created cart for call session: {cart}")

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
        agent=Assistant(
            chat_ctx=ChatContext(), 
            participant_identity=participant.identity, 
            room_name=ctx.room.name,
            cart=cart
        ),
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