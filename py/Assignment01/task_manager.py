# task_manager.py

import json
import os

# File to store tasks persistently
task_file = "tasks.json"

# Load tasks from file if it exists
if os.path.exists(task_file):
    with open(task_file, "r") as file:
        tasks = json.load(file)
else:
    tasks = []

def save_tasks():
    """Saves the current tasks list to a JSON file."""
    with open(task_file, "w") as file:
        json.dump(tasks, file, indent=2)

def add_task(description):
    """Adds a new task with the given description."""
    task_id = tasks[-1]["id"] + 1 if tasks else 1
    task = {"id": task_id, "description": description, "completed": False}
    tasks.append(task)
    save_tasks()
    print(f"Task '{description}' added with ID {task_id}.")

def view_tasks():
    """Displays all tasks with their status."""
    if not tasks:
        print("No tasks available.")
        return
    for task in tasks:
        status = "Done" if task["completed"] else "Pending"
        print(f"{task['id']}: {task['description']} [{status}]")

def mark_completed(task_id):
    """Marks the task with the given ID as completed."""
    for task in tasks:
        if task["id"] == task_id:
            task["completed"] = True
            save_tasks()
            print(f"Task ID {task_id} marked as completed.")
            return
    print(f"No task found with ID {task_id}.")

def delete_task(task_id):
    """Deletes the task with the given ID."""
    global tasks
    new_tasks = [task for task in tasks if task["id"] != task_id]
    if len(new_tasks) == len(tasks):
        print(f"No task found with ID {task_id}.")
    else:
        tasks[:] = new_tasks
        save_tasks()
        print(f"Task ID {task_id} deleted.")

def main(commands):
    """Main function to process a list of commands."""
    for command in commands:
        choice = command[0]
        if choice == "add":
            if len(command) > 1:
                add_task(command[1])
            else:
                print("Add command requires a description.")
        elif choice == "view":
            view_tasks()
        elif choice == "complete":
            if len(command) > 1:
                try:
                    task_id = int(command[1])
                    mark_completed(task_id)
                except ValueError:
                    print("Invalid task ID.")
            else:
                print("Complete command requires a task ID.")
        elif choice == "delete":
            if len(command) > 1:
                try:
                    task_id = int(command[1])
                    delete_task(task_id)
                except ValueError:
                    print("Invalid task ID.")
            else:
                print("Delete command requires a task ID.")
        elif choice == "exit":
            print("Exiting Task Manager. Goodbye!")
            break
        else:
            print(f"Invalid command: {choice}")

# Sample usage with automated commands
if __name__ == "__main__":
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
    main(commands_to_execute)
