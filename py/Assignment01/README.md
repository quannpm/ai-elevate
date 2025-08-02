# 🗂️ Persistent Task Manager (Python + JSON)

This is a simple task manager built in Python that stores your tasks **persistently** using a local `tasks.json` file.

It supports the following operations:
- ✅ Add a task
- 👀 View tasks
- ☑️ Mark tasks as completed
- 🗑️ Delete tasks
- 💾 Auto-save tasks between runs

---

## 📂 Project Files

```
task_manager.py     # Main Python script
tasks.json          # Auto-generated file to store tasks
README.md           # You're reading this :)
```

---

## 🚀 How It Works

- When the script starts, it checks if `tasks.json` exists.
- If it does, tasks are loaded from that file.
- When tasks are added, completed, or deleted, the list is automatically saved back to `tasks.json`.

Each task is stored as a dictionary:
```json
{
  "id": 1,
  "description": "Walk the dog",
  "completed": false
}
```

---

## ▶️ Example Commands

The script uses a predefined list of commands in the form of tuples.

```python
commands_to_execute = [
    ("add", "Buy groceries"),
    ("add", "Walk the dog"),
    ("view",),
    ("complete", "1"),
    ("view",),
    ("delete", "2"),
    ("view",),
    ("exit",)
]
```

You can modify the `commands_to_execute` in the script for testing different actions.

---

## 🛠 Main Functions

| Function             | Description                                    |
|----------------------|------------------------------------------------|
| `add_task()`         | Adds a task to the list                        |
| `view_tasks()`       | Displays all tasks and their statuses          |
| `mark_completed()`   | Marks a task as completed                      |
| `delete_task()`      | Deletes a task by ID                           |
| `save_tasks()`       | Saves the tasks to `tasks.json`                |
| `main()`             | Processes commands from a list                 |

---

## 💡 Notes

- Task IDs are unique and auto-incremented.
- Tasks are saved after **every action**.
- This is a beginner-friendly project for learning Python, JSON, and file handling.
- You can extend it to support real-time `input()` or build a CLI with `argparse`.

---

## ▶️ Running the Project

Save the script as `task_manager.py`, then run it:

```bash
python task_manager.py
```

A `tasks.json` file will be created/updated in the same directory.

---

## ❤️ from QuanNPM1 with love for learning AI.