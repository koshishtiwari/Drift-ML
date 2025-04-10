import os
import asyncio
import json
import logging
import nest_asyncio
from alpaca.data.live import StockDataStream

# Patch asyncio to allow nested event loops.
nest_asyncio.apply()

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve Alpaca API credentials from environment variables.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK1KCLI42AU3V12T1ZP1")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "hXgs5buBHjfUwoNDhsWSuNkhwGDdztLkRNf6noKI")


def serialize_data(data):
    """
    Attempts to convert a non-serializable object (like a Quote) to a dictionary.
    If the object has a __dict__ attribute, it returns that; otherwise, returns the string representation.
    """
    try:
        return data.__dict__
    except AttributeError:
        return str(data)

async def quote_data_handler(data):
    """
    Async callback invoked when a real-time quote is received.
    Serializes the quote object (including datetime objects) into JSON-friendly format,
    logs its content, and sends it as a prompt to Gemini LLM for analysis.
    Adds a delay to slow down the data processing.
    """
    serializable_data = serialize_data(data)
    
    # Use a custom default lambda: if an object has isoformat, use that (for datetime), else str(o)
    json_data = json.dumps(
        serializable_data,
        indent=2,
        default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)
    )
    
    logger.info("Received Quote Data:\n%s", json_data)
    
    # Introduce a delay (e.g., 1 second) to slow down the processing.
    await asyncio.sleep(1)

async def run_alpaca_stream():
    """
    Sets up the Alpaca live data stream, subscribes to real-time SPY quotes,
    and initiates the WebSocket connection.
    """
    logger.info("Setting up Alpaca StockDataStream...")

    # Create the live data stream instance.
    wss_client = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # Subscribe to real-time quote data for SPY.
    wss_client.subscribe_quotes(quote_data_handler, "SPY")
    logger.info("Subscription to SPY quotes established. Starting stream...")

    # This call runs indefinitely until interrupted.
    await wss_client.run()

async def main():
    # Run Alpaca live data stream.
    await run_alpaca_stream()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stream interrupted and closed.")
