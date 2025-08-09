import os
import openai

# Step 1: Dummy input logs
log_entries = [
    "Driver reported heavy traffic on highway due to construction",
    "Package not accepted, customer unavailable at given time",
    "Vehicle engine failed during route, replacement dispatched",
    "Unexpected rainstorm delayed loading at warehouse",
    "Sorting label missing, required manual barcode scan",
    "Driver took a wrong turn and had to reroute",
    "No issue reported, arrived on time",
    "Address was incorrect, customer unreachable",
    "System glitch during check-in at loading dock",
    "Road accident caused a long halt near delivery point"
]

# Step 2: Heuristic Pre-classifier
def initial_classify(text):
    keywords = {
        "traffic": "Traffic",
        "road accident": "Traffic",
        "customer": "Customer Issue",
        "unavailable": "Customer Issue",
        "engine": "Vehicle Issue",
        "vehicle": "Vehicle Issue",
        "rain": "Weather",
        "storm": "Weather",
        "label": "Sorting/Labeling Error",
        "barcode": "Sorting/Labeling Error",
        "wrong turn": "Human Error",
        "reroute": "Human Error",
        "system": "Technical System Failure",
        "glitch": "Technical System Failure"
    }

    text = text.lower()
    for k, v in keywords.items():
        if k in text:
            return v
    return "Other"

# Step 3: OpenAI Client
client = openai.OpenAI(
    base_url="https://aiportalapi.stu-platform.live/jpe",
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Step 4: GPT-Based Refinement Layer
def refine_classification(text, initial_label):
    prompt = f"""
You are a logistics assistant. A log entry has been auto-categorized as "{initial_label}".
Please confirm or correct it by choosing one of the following categories:

- Traffic
- Customer Issue
- Vehicle Issue
- Weather
- Sorting/Labeling Error
- Human Error
- Technical System Failure
- Other

Log Entry:
```{text}```

Return only the most appropriate category from the list.
"""
    response = client.chat.completions.create(
        model="GPT-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Step 5: Pipeline
def classify_log(text):
    initial = initial_classify(text)
    final = refine_classification(text, initial)
    return {"log": text, "initial": initial, "final": final}

# Step 6: Run All
if __name__ == "__main__":
    print("Delay Reason Classification Results:\n")
    results = []
    for entry in log_entries:
        result = classify_log(entry)
        results.append(result)
        print(f"Log: {result['log']}")
        print(f"Initial Prediction: {result['initial']}")
        print(f"Final Category: {result['final']}\n")

    # Summary table
    print("\n=== Final Prediction Summary ===")
    for r in results:
        print(f"- {r['log']} → {r['final']}")

#Summary Final Submission Checklist
"""
This classification system combines a heuristic keyword-based classifier with a refinement layer powered by OpenAI. 
Unlike retrieval-based pipelines (which depend on a static database of examples or embeddings), this system dynamically reasons about unseen text using large language models. 
It's best suited for tasks with varied, unstructured input and where new log patterns appear frequently, making retrieval methods less effective.
"""