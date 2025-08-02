import os
import time
import json
from openai import RateLimitError, APIError
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# Configure Azure OpenAI Client
client = openai.OpenAI(
    base_url="https://aiportalapi.stu-platform.live/jpe",
    api_key="sk-89XCTFT0--byUic6zSnFTw"
)

# Function schema for function calling
functions = [
    {
        "name": "generate_itinerary",
        "description": "Generate a travel itinerary for a given destination and duration.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "Travel destination city or country"},
                "days": {"type": "integer", "description": "Number of days to plan for"},
                "itinerary": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of daily activities with output format Day number: activity"
                }
            },
            "required": ["destination", "days"],
        },
    }
]

# Retry logic for API call
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
def call_openai_function(prompt, destination, days):
    response = client.chat.completions.create(
        model="GPT-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        functions=functions,
        function_call={
            "name": "generate_itinerary",
            "arguments": f'{{"destination": "{destination}", "days": {days}}}'
        }
    )
    return response

# Batch processing
def batch_process(inputs):
    results = []
    for input_data in inputs:
        prompt = input_data["prompt"]
        destination = input_data["destination"]
        days = input_data["days"]
        try:
            print(f"Processing: {destination} for {days} days")
            response = call_openai_function(prompt, destination, days)
            func_call = response.choices[0].message.function_call
            args = json.loads(func_call.arguments)
            itinerary = args.get("itinerary", [])
            results.append({
                "destination": destination,
                "days": days,
                "itinerary": itinerary
            })
        except Exception as e:
            print(f"❌ Error processing {destination}: {e}")
            results.append({
                "destination": destination,
                "days": days,
                "itinerary": ["No result due to error."]
            })
        print("-" * 50)
        time.sleep(1)  # Prevent aggressive requests
    return results

# Dummy input batch
batch_inputs = [
    {"prompt": "Plan a travel itinerary.", "destination": "Paris", "days": 3},
    {"prompt": "Plan a travel itinerary.", "destination": "Tokyo", "days": 5},
    {"prompt": "Plan a travel itinerary.", "destination": "New York", "days": 4},
]

# Run and show results
if __name__ == "__main__":
    results = batch_process(batch_inputs)
    print("\n=== Batch Results ===")
    for item in results:
        print(f"\nDestination: {item['destination']} in {item['days']} days")
        print("Plan:")
        if not item["itinerary"]:
            print("No plan available")
        else:
            for activity in item["itinerary"]:
                print(f"  {activity}")
        print("=" * 60)
