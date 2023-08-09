import sounddevice as sd
import soundfile as sf
import pyttsx3
import tempfile
import os
import openai
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, load_tools
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, Tool
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.utilities import GoogleSearchAPIWrapper, GoogleSerperAPIWrapper

OPENAI_API_KEY =  ""
ZAPIER_NLA_API_KEY = ""
GOOGLE_API_KEY = ""
GOOGLE_CSE_ID = ""
SERPER_API_KEY = ""
# Set recording parameters
duration = 5  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels

def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Finished recording.")
    return recording

def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()

        # Use OpenAI API for audio transcription
        # Make sure to set your OPENAI_API_KEY before using this
        openai.api_key = OPENAI_API_KEY
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        os.remove(temp_audio.name)

    return transcript["text"].strip()


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize Langchain components
llm = OpenAI(temperature=0, openai_api_key= OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history")
zapier = ZapierNLAWrapper(zapier_nla_api_key = ZAPIER_NLA_API_KEY)
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

# make custom tools
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class GoogleSearchTool(BaseTool):
    name = "Google Search"
    description = "Use this tool to search on Google."

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Google Search does not support async")


class GoogleSerperTool(BaseTool):
    name = "Google Serper"
    description = "Useful for when you need to ask with search"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
        return search.run(query)

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Google Serper does not support async")

tools = [
    GoogleSearchTool(),
    GoogleSerperTool(),
] + toolkit.get_tools() + load_tools(["human"])

agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)


while True:
    print("Press Enter to start recording.")
    input()  # Wait for Enter key
    recorded_audio = record_audio(duration, fs, channels)
    message = transcribe_audio(recorded_audio, fs)
    print(f"You: {message}")
    assistant_message = agent.run(message)
    speak(assistant_message)