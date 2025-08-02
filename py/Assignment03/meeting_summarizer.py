import os
import argparse
import openai

def load_transcript(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_summary(transcript):
    client = openai.OpenAI(
        base_url="https://aiportalapi.stu-platform.live/jpe",
        api_key="YOUR_API_KEY"
    )

    prompt = (
        "Summarize the following meeting transcript with key points, decisions, and action items:\n\n"
        f"{transcript}"
    )

    response = client.chat.completions.create(
        model="GPT-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in summarizing meeting notes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

def main():
    print("=== AI-Powered Meeting Summarizer ===")
    print("Enter the full path or just the file name (in current directory).")
    input_path = input("Transcript file path: ").strip()

    try:
        transcript = load_transcript(input_path)
        summary = generate_summary(transcript)
        print("\n--- Meeting Summary ---\n")
        print(summary)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
