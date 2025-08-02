from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set environment variables if not already set
# Uncomment the following lines to set them manually if needed
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-azure-openai-endpoint"
# os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
# Ensure the environment variables are set
# os.environ["AZURE_OPENAI_ENDPOINT"] = ""
# os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_DEPLOYMENT_NAME"] = "GPT-4o-mini"
if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
    raise EnvironmentError("Please set the AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Few-shot examples for sentiment analysis
few_shot_examples = [
    {"role": "user", "content": "Analyze the sentiment of this text: 'I love attending networking events!'"},
    {"role": "assistant", "content": "Sentiment: Positive. The text expresses enthusiasm and enjoyment."},
    {"role": "user", "content": "Analyze the sentiment of this text: 'Networking can be really stressful sometimes.'"},
    {"role": "assistant", "content": "Sentiment: Negative. The text shows discomfort and stress related to networking."},
]

# System message
conversation_messages = [
    {"role": "system", "content": "You are a helpful event management assistant."}
]

# Append few-shot examples
conversation_messages.extend(few_shot_examples)

# Add user questions with chain-of-thought prompting
user_questions = [
    {
        "role": "user",
        "content": (
            "What are some good conversation starters at networking events? "
            "Explain your reasoning step-by-step."
        )
    },
    {
        "role": "user",
        "content": (
            "How can I overcome nervousness when meeting new people at events? "
            "Please explain your reasoning step-by-step."
        )
    },
    {
        "role": "user",
        "content": (
            "Suggest ways to follow up after meeting someone at a networking event. "
            "Explain your reasoning step-by-step."
        )
    }
]

for question in user_questions:
    # Preserve context by copying the conversation history
    messages = conversation_messages.copy()
    messages.append(question)

    # Call Azure OpenAI chat completion
    response = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    # Extract and print assistant reply
    assistant_reply = response.choices[0].message.content
    print("="*60)
    print(f"User: {question['content']}\n")
    print("Assistant Response:\n")
    print(assistant_reply)
    print("="*60 + "\n")
