import os
import csv
import openai

# Step 1: Mock Input Data
task_descriptions = [
    "Install the battery module in the rear compartment, connect to the high-voltage harness, and verify torque on fasteners.",
    "Calibrate the ADAS (Advanced Driver Assistance Systems) radar sensors on the front bumper using factory alignment targets.",
    "Apply anti-corrosion sealant to all exposed welds on the door panels before painting.",
    "Perform leak test on coolant system after radiator installation. Record pressure readings and verify against specifications.",
    "Program the infotainment ECU with the latest software package and validate connectivity with dashboard display."
]

# Step 2: OpenAI Setup
client = openai.OpenAI(
        base_url="https://aiportalapi.stu-platform.live/jpe",
        api_key="YOUR_API_KEY"
)

# Step 3: Prompt to Generate Instructions
def generate_instruction(task):
    prompt = f"""
You are an expert automotive manufacturing supervisor. Generate step-by-step work instructions for the following new model task.
Include safety precautions, required tools (if any), and acceptance checks. Write in clear, numbered steps suitable for production workers.

Task:
\"\"\"{task}\"\"\"

Work Instructions:
"""
    response = client.chat.completions.create(
        model="GPT-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Step 4: Generate and Save to CSV
output_file = "output_instructions.csv"
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Task Description", "Generated Work Instructions"])

    for task in task_descriptions:
        instructions = generate_instruction(task)
        writer.writerow([task, instructions])
        print(f"Generated instructions for task:\n{task}\n")

print(f"\nAll instructions saved to '{output_file}' successfully.")
