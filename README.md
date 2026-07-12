# Teena

Teena is a personal Telegram assistant bot built for personal use. It is designed to help manage tasks, sync with Google Calendar, and chat using an LLM (Gemini) for advice and daily planning. This is built as a learning project and is currently in active development.

## Features

### Currently Implemented
* **Task Management**:
  * `/addtask <task text>` - Add a new task to your list.
  * `/tasks` - Display all open/incomplete tasks.
  * `/done <id>` - Mark a task as completed.
* **Google Calendar Integration**:
  * `/agenda` - View today's Google Calendar events.
  * Schedule new calendar events via natural language (e.g. "Schedule a team meeting tomorrow at 3pm").
* **LLM Chat (Gemini)**:
  * Conversational AI that understands your tasks and calendar context.
  * Natural-language actions — add tasks, complete tasks, and create events just by chatting (no slash commands needed).
* **SQLite-based Storage**: Locally saves tasks, conversation history, and metadata.

### Planned / In Progress
* **Mood Tracking**: Log and visualize daily mood entries.
* **Long-Term Memory**: Persistent personal facts/preferences the bot learns over time.
* **Proactive Scheduled Check-ins**: Automated messages asking for daily check-ins or reminding about due tasks.

## Tech Stack
* **Python 3.12**
* **python-telegram-bot v21.x** (async style)
* **SQLite** (standard library database)
* **Google Calendar API** (OAuth 2.0, read/write)
* **Google Gemini API** (gemini-3.1-flash-lite for chat and intent detection)
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
   Create a file named `.env` in the root of the project and add:
   ```env
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   * You can get a Telegram bot token from [@BotFather](https://t.me/BotFather).
   * You can get a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

6. **Set up Google Calendar credentials**:
   * Go to the [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials.
   * Create an OAuth 2.0 Client ID (Desktop application type).
   * Download the JSON file and save it as `credentials.json` in the project root.
   * On first run, the bot will open a browser window for you to authorize calendar access. The resulting token is saved to `token.json` for future runs.

7. **Run the bot**:
   ```bash
   python main.py
   ```

## Project Structure

* **[main.py](main.py)**: Entry point — Telegram command handlers (`/start`, `/addtask`, `/tasks`, `/done`, `/agenda`), the natural-language chat handler with intent routing, and the polling loop.
* **[database.py](database.py)**: SQLite database setup, table creation (tasks, messages, mood_logs, facts), and helper functions for task and message operations.
* **[llm_helper.py](llm_helper.py)**: Gemini-powered intent detection (`detect_intent()`) and conversational reply generation (`generate_reply()`), including prompt building with task/calendar/history context.
* **[calendar_helper.py](calendar_helper.py)**: Google Calendar API integration — OAuth authentication, fetching today's events (`get_today_events()`), and creating new events (`create_event()`).

## Status / Roadmap

This is a personal work-in-progress project built incrementally across multiple phases:

* **Phase 1 (Done)**: Skeleton bot with basic message echo and polling setup.
* **Phase 2 (Done)**: SQLite task management with `/addtask`, `/tasks`, `/done`.
* **Phase 3 (Done)**: Google Calendar integration — `/agenda` to view today's events.
* **Phase 4a (Done)**: Gemini LLM chat — context-aware conversational replies.
* **Phase 4b (Done)**: Natural-language intent detection — add tasks and complete tasks via free text.
* **Phase 4c (Done)**: Natural-language event creation — schedule calendar events via free text.
* **Phase 4d (Next)**: Reschedule/delete support for tasks and events; conversation-aware intent detection.
* **Future Phases**:
  * Mood tracking feature
  * Long-term memory (facts/preferences)
  * Proactive/scheduled check-ins using APScheduler
