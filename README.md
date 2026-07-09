# Teena

Teena is a personal Telegram assistant bot built for personal use. It is designed to help manage tasks, track mood, sync with Google Calendar, and chat using an LLM (Gemini) for advice and daily planning. This is built as a learning project and is currently in active development.

## Features

### Currently Implemented
* **Task Management**:
  * `/addtask <task text>` - Add a new task to your list.
  * `/tasks` - Display all open/incomplete tasks.
  * `/done <id>` - Mark a task as completed.
* **SQLite-based Storage**: Locally saves all task information.

### Planned / In Progress
* **Google Calendar Sync**: Syncing tasks and events directly to a calendar.
* **Mood Tracking**: Log and visualize daily mood entries.
* **LLM Chat**: Conversational AI utilizing Google Gemini for scheduling, planning, and general advice.
* **Proactive Scheduled Check-ins**: Automated messages asking for daily check-ins or reminding about due tasks.

## Tech Stack
* **Python 3.12**
* **python-telegram-bot v21.x** (async style)
* **SQLite** (standard library database)
* **Google Calendar API** (planned)
* **Google Gemini API** (planned)
* **APScheduler** (planned for check-ins)

## Setup Instructions

Follow these steps to set up and run the bot locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Joshua-roland-williams/teena-pa.git
   cd teena-pa
   ```

2. **Create a virtual environment**:
   ```bash
   py -3.12 -m venv venv
   ```

3. **Activate the virtual environment**:
   * **Windows**:
     ```cmd
     venv\Scripts\activate
     ```
   * **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**:
   Create a file named `.env` in the root of the project and add your Telegram bot token:
   ```env
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   ```
   *(Note: You can get a bot token from [@BotFather](https://t.me/BotFather) on Telegram).*

6. **Run the bot**:
   ```bash
   python main.py
   ```

## Project Structure

* **[main.py](file:///c:/Users/LENOVO/Desktop/TBOT/main.py)**: Contains the Telegram bot logic, command handlers (e.g., `/start`, `/addtask`, `/tasks`, `/done`), and the polling loop.
* **[database.py](file:///c:/Users/LENOVO/Desktop/TBOT/database.py)**: Handles direct SQLite database connections, table creation (`tasks`, `mood_logs`, etc.), and helper functions for CRUD operations.

## Status / Roadmap

This is a personal work-in-progress project built incrementally across multiple phases:

* **Phase 1 (Done)**: Skeleton bot implementation with basic message echo and polling setup.
* **Phase 2 (Done)**: SQLite task management with interactive commands.
* **Phase 3 (Next)**: Google Calendar integration to sync tasks.
* **Future Phases**:
  * Mood tracking feature
  * LLM chat integration with Google Gemini
  * Proactive/scheduled check-ins using APScheduler
