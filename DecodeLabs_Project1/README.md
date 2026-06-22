# Rule-Based AI Chatbot

A very simple Python console application that simulates an AI chatbot using basic rule-based logic. It relies entirely on `if-elif-else` statements to process user inputs and formulate responses. This makes it an excellent, beginner-friendly first-year programming project.

## Features
- Runs in a continuous loop until the user exits.
- Case-insensitive input handling.
- Purely conditional logic (no AI libraries, machine learning, or APIs).
- Predefined responses for 10+ specific commands.
- Graceful handling of unknown inputs.

## Available Commands
- `hi` / `hello` -> Greet the user
- `how are you` -> Respond politely
- `name` -> Tell the bot's name
- `age` -> Tell the bot's age
- `joke` -> Tell a programming joke
- `fact` -> Share a fun fact
- `weather` -> Comment on the weather
- `time` -> Comment on the time
- `hobby` -> Share the bot's hobby
- `help` -> Display available commands
- `bye` / `exit` -> Print a goodbye message and terminate the program

## Requirements
- Python 3.x
- **No external libraries are required.** The application is built entirely with Python's standard built-in functions.

## How to Run
1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run the script using Python:
   ```bash
   python chatbot.py
   ```
4. Start chatting by typing your commands!
