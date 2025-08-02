# Batch Travel Itinerary Generator (OpenAI)

This Python module demonstrates efficient use of the OpenAI API with:
- Function calling
- Batching
- Robust retry logic (with `tenacity`)
- Graceful error handling

---

## 🔧 Requirements

- Python 3.7+
- `openai` SDK
- `tenacity` library

Install required libraries:
```bash
pip3 install openai tenacity
```

---

## 🚀 How to Run

1. Run the script:
```bash
python batch_itinerary_generator.py
```

---

## 📦 Features

- **Function Calling**:
  Uses OpenAI's function calling to generate structured travel itineraries.

- **Batch Processing**:
  Accepts multiple destinations in one run to maximize API usage efficiency.

- **Retry Logic**:
  Uses `tenacity` to retry on API rate limits and transient errors (up to 5 times with exponential backoff).

- **Error Handling**:
  Each input is processed independently; errors are caught and logged without crashing.

---

## 🧪 Sample Input (in code)

```python
batch_inputs = [
    {"prompt": "Plan a travel itinerary.", "destination": "Paris", "days": 3},
    {"prompt": "Plan a travel itinerary.", "destination": "Tokyo", "days": 5},
    {"prompt": "Plan a travel itinerary.", "destination": "New York", "days": 4},
]
```

---

## 📈 Output

The script prints and returns a structured itinerary or an error message per destination.
