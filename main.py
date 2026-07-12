"""
main.py — Teena, your Telegram personal assistant bot.

This is the entry point for the bot. It handles:
  • /start    → sends a welcome message
  • /addtask  → adds a new task to the database
  • /tasks    → lists all open (incomplete) tasks
  • /done     → marks a task as completed by its id
  • /agenda   → shows today's Google Calendar events
  • Any text  → Gemini-powered intent detection first (can add tasks,
                 complete tasks, or schedule calendar events via natural
                 language), then falls back to conversational chat.

Uses python-telegram-bot v21.x (async style) with polling.
"""

import datetime
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Import our database helpers from database.py
from database import init_db, add_task, get_open_tasks, mark_task_done, save_message, get_recent_messages

# Import the Google Calendar helper for the /agenda command, event creation,
# and upcoming-week context for chat replies
from calendar_helper import get_today_events, get_upcoming_events, create_event

# Import the Gemini LLM helper for conversational chat and intent detection
from llm_helper import generate_reply, detect_intent

# ---------------------------------------------------------------------------
# 1. Load environment variables from .env
# ---------------------------------------------------------------------------
load_dotenv()  # reads the .env file in the project root
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Quick sanity check — crash early if the token is missing
if not BOT_TOKEN:
    raise ValueError(
        "TELEGRAM_BOT_TOKEN is not set! "
        "Make sure your .env file contains it."
    )

# ---------------------------------------------------------------------------
# 2. Set up logging so we can see what's happening in the terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 3. Define command / message handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command — greet the user."""
    user_first_name = update.effective_user.first_name
    welcome_text = (
        f"Hey {user_first_name}! 👋\n\n"
        "I'm *Teena*, your personal assistant bot.\n\n"
        "Here's what I can do so far:\n"
        "• `/addtask <text>` — add a task\n"
        "• `/tasks` — see your open tasks\n"
        "• `/done <id>` — mark a task complete\n"
        "• `/agenda` — see today's calendar events\n\n"
        "Or just send me a message — I can add tasks, schedule "
        "events, and chat! 💬"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")
    logger.info("Sent welcome message to %s", user_first_name)


# ---------------------------------------------------------------------------
# Task management command handlers
# ---------------------------------------------------------------------------

# NOTE: This command only sets the task text.  Priority, category, and
# due_date are intentionally settable only via natural language (the
# detect_intent() path in chat()), not through this slash command.
async def addtask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /addtask <task text> — create a new task.

    `context.args` is a list of words that follow the command.
    For example, "/addtask Buy groceries" gives context.args = ["Buy", "groceries"].
    We join them back into a single string.
    """

    # If the user just types "/addtask" with nothing after it, guide them.
    if not context.args:
        await update.message.reply_text(
            "Usage: `/addtask <task text>`\n"
            "Example: `/addtask Buy groceries`",
            parse_mode="Markdown",
        )
        return

    # Join all the words back into one task description string
    task_text = " ".join(context.args)

    # Save to the database and get the new task's id
    task_id = add_task(task_text)

    await update.message.reply_text(
        f"Got it! Task added.\n"
        f"ID: `{task_id}`\n"
        f"Task: {task_text}",
        parse_mode="Markdown",
    )
    logger.info("User %s added task #%d: %s", update.effective_user.first_name, task_id, task_text)


async def tasks_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /tasks — list all open (incomplete) tasks.

    Calls get_open_tasks() which returns a list of dicts,
    each with keys: id, text, due_date, done, created_at,
    priority, category, completed_at.
    """

    open_tasks = get_open_tasks()

    # If there's nothing to show, send a friendly message instead of an empty list
    if not open_tasks:
        await update.message.reply_text(
            "You have no open tasks — enjoy your free time! 🎉"
        )
        return

    # Build a numbered list, showing each task's id and text.
    # If a due date is set, append it in parentheses.
    lines = ["📋 *Your open tasks:*\n"]
    for i, task in enumerate(open_tasks, start=1):
        due_info = f"  _(due {task['due_date']})_" if task["due_date"] else ""
        lines.append(f"{i}. [`{task['id']}`] {task['text']}{due_info}")

    # Add a tip at the bottom so the user knows how to complete tasks
    lines.append("\nMark one done with `/done <id>`")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    logger.info("Listed %d open tasks for %s", len(open_tasks), update.effective_user.first_name)


async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /done <task_id> — mark a task as completed.

    We use try/except to handle the case where the user passes
    something that isn't a valid number (e.g. "/done abc").
    """

    # If the user just types "/done" with no argument, guide them.
    if not context.args:
        await update.message.reply_text(
            "Usage: `/done <task_id>`\n"
            "Example: `/done 3`\n\n"
            "Use `/tasks` to see your open tasks and their IDs.",
            parse_mode="Markdown",
        )
        return

    # Try to convert the first argument to an integer.
    # If the user types something like "/done abc", this will fail.
    try:
        task_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text(
            f"`{context.args[0]}` is not a valid task ID.\n"
            "Please provide a number, e.g. `/done 3`",
            parse_mode="Markdown",
        )
        return

    # Attempt to mark the task as done in the database
    if mark_task_done(task_id):
        await update.message.reply_text(f"Task `{task_id}` marked as done! ✅", parse_mode="Markdown")
        logger.info("User %s completed task #%d", update.effective_user.first_name, task_id)
    else:
        await update.message.reply_text(
            f"No open task found with ID `{task_id}`.\n"
            "Use `/tasks` to see your current tasks.",
            parse_mode="Markdown",
        )


# ---------------------------------------------------------------------------
# Calendar / agenda command handler
# ---------------------------------------------------------------------------

async def agenda_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /agenda — show today's Google Calendar events.

    Calls get_today_events() which returns a list of dicts with keys:
    summary, start, end.  Wraps the call in try/except so that
    credential issues or API errors never crash the bot.
    """

    try:
        events = get_today_events()
    except FileNotFoundError as exc:
        # credentials.json is missing — tell the user, log the real error
        logger.error("Calendar credentials missing: %s", exc)
        await update.message.reply_text(
            "⚠️ Calendar credentials are not set up yet.\n"
            "Please add your `credentials.json` file and try again.",
            parse_mode="Markdown",
        )
        return
    except Exception as exc:
        # Any other Google API or network error
        logger.error("Failed to fetch calendar events: %s", exc, exc_info=True)
        await update.message.reply_text(
            "⚠️ Sorry, I couldn't fetch your calendar right now.\n"
            "Please try again later.",
        )
        return

    # No events today
    if not events:
        await update.message.reply_text(
            "You have no events today — clear schedule! 🎉"
        )
        logger.info("No calendar events today for %s", update.effective_user.first_name)
        return

    # Build a formatted agenda list
    lines = ["📅 *Today's agenda:*\n"]
    for event in events:
        # All-day events show "All day" instead of a time range
        if event["start"] == "All day":
            lines.append(f"  🔹 {event['summary']}  _(all day)_")
        else:
            lines.append(f"  🔹 {event['start']} – {event['end']}  {event['summary']}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    logger.info(
        "Showed %d calendar event(s) to %s",
        len(events),
        update.effective_user.first_name,
    )


# ---------------------------------------------------------------------------
# Chat handler — Gemini-powered conversational replies
# ---------------------------------------------------------------------------

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any plain text message — first attempt structured intent detection,
    then fall back to Gemini-powered conversational chat.

    Flow:
      1. Save the user's message to the database.
      2. Gather context: open tasks, today's calendar, recent history.
      3. Call detect_intent() to classify the message as an action or chat.
      4. Branch on the detected intent:
         • add_task       → create the task in the DB, reply with confirmation.
         • complete_task   → validate & mark done, reply with confirmation.
         • add_event       → parse date/time, create a Google Calendar event.
         • chat (default)  → call generate_reply() for a conversational answer.
      5. Save the assistant's reply to the database.
    """
    user_text = update.message.text
    user_name = update.effective_user.first_name

    # Step 1 — Persist the incoming user message
    save_message("user", user_text)

    # Step 2 — Gather context for the LLM
    open_tasks = get_open_tasks()
    recent_messages = get_recent_messages(limit=10)

    # Step 3 — Detect the user's intent before falling back to normal chat.
    # detect_intent() always returns a dict with at least {"intent": "chat"}
    # so it's safe to branch on without extra null checks.
    intent_data = detect_intent(user_text, open_tasks)
    intent = intent_data.get("intent", "chat")

    # ------------------------------------------------------------------
    # Step 4 — Branch on the detected intent
    # ------------------------------------------------------------------

    if intent == "add_task":
        # ---- ADD TASK intent ----
        # Extract the structured fields Gemini returned.  task_text is
        # required; priority defaults to "medium", category and due_date
        # default to None — matching the database.add_task() signature.
        task_text = intent_data.get("task_text", user_text)
        priority = intent_data.get("priority", "medium")
        category = intent_data.get("category")
        due_date = intent_data.get("due_date")

        # Persist the new task in the database
        new_id = add_task(task_text, due_date=due_date, priority=priority, category=category)

        # Build a natural confirmation message, only mentioning non-default
        # / non-null fields so it doesn't feel robotic.
        extras = []
        if priority and priority != "medium":
            extras.append(f"priority *{priority}*")
        if category:
            extras.append(f"category *{category}*")
        if due_date:
            extras.append(f"due *{due_date}*")

        extra_str = (", ".join(extras))
        if extra_str:
            reply = f"✅ Added task: *{task_text}* ({extra_str})"
        else:
            reply = f"✅ Added task: *{task_text}*"

        await update.message.reply_text(reply, parse_mode="Markdown")
        save_message("assistant", reply)
        logger.info(
            "Intent add_task from %s — created task #%d: %s",
            user_name, new_id, task_text,
        )

    elif intent == "complete_task":
        # ---- COMPLETE TASK intent ----
        # Gemini returns a task_id it thinks the user is referring to.
        # We validate it against the open_tasks list we already fetched
        # to guard against hallucinated or stale ids.
        target_id = intent_data.get("task_id")

        # Build a set of valid open task ids for fast lookup
        open_ids = {t["id"] for t in open_tasks}

        if target_id is not None and target_id in open_ids:
            # ID is valid — mark it done
            mark_task_done(target_id)

            # Find the task text so we can confirm by name, not just id
            task_name = next(
                (t["text"] for t in open_tasks if t["id"] == target_id),
                f"task #{target_id}",
            )
            reply = f"✅ Done! Marked *{task_name}* as complete."
            await update.message.reply_text(reply, parse_mode="Markdown")
            save_message("assistant", reply)
            logger.info(
                "Intent complete_task from %s — completed task #%d: %s",
                user_name, target_id, task_name,
            )
        else:
            # ID is missing, invalid, or not in the open tasks list.
            # Don't silently complete the wrong task — show the list
            # so the user can clarify.
            lines = ["🤔 I wasn't sure which task you meant. Here are your open tasks:\n"]
            if open_tasks:
                for t in open_tasks:
                    due = f"  _(due {t['due_date']})_" if t.get("due_date") else ""
                    lines.append(f"  • [`{t['id']}`] {t['text']}{due}")
                lines.append("\nYou can say which one, or use `/done <id>`.")
            else:
                lines.append("  (You have no open tasks right now.)")

            reply = "\n".join(lines)
            await update.message.reply_text(reply, parse_mode="Markdown")
            save_message("assistant", reply)
            logger.info(
                "Intent complete_task from %s — could not match task_id=%s",
                user_name, target_id,
            )
    elif intent == "add_event":
        # ---- ADD EVENT intent ----
        # Gemini extracted: summary, date (YYYY-MM-DD), start_time (HH:MM).
        # We parse these into timezone-aware datetimes and default to a
        # 1-hour duration since the intent doesn't include an end time.
        event_summary = intent_data.get("summary", "New event")
        date_str = intent_data.get("date")        # e.g. "2026-07-13"
        time_str = intent_data.get("start_time")  # e.g. "15:00"

        # Both date and time are required — if either is missing, the
        # intent was probably incomplete.  Fall through to chat so Teena
        # can ask the user for the missing details.
        if not date_str or not time_str:
            reply = generate_reply(
                user_message=user_text,
                open_tasks=open_tasks,
                today_events=[],
                recent_messages=recent_messages,
            )
            await update.message.reply_text(reply)
            save_message("assistant", reply)
            logger.info(
                "Intent add_event from %s — missing date/time, fell back to chat",
                user_name,
            )
        else:
            try:
                # Parse the date and time strings into a naive datetime,
                # then make it timezone-aware using the system's local tz.
                naive_start = datetime.datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
                )
                # .astimezone() with no arg uses the system's local timezone
                local_tz = datetime.datetime.now().astimezone().tzinfo
                start_dt = naive_start.replace(tzinfo=local_tz)

                # Default duration: 1 hour
                end_dt = start_dt + datetime.timedelta(hours=1)

                # Create the event via Google Calendar API
                event_id = create_event(event_summary, start_dt, end_dt)

                # Format the time for a human-friendly confirmation
                friendly_time = start_dt.strftime("%I:%M %p").lstrip("0")
                friendly_date = start_dt.strftime("%A, %B %d")
                reply = (
                    f"📅 Scheduled: *{event_summary}*\n"
                    f"{friendly_date} at {friendly_time} (1 hour)"
                )
                await update.message.reply_text(reply, parse_mode="Markdown")
                save_message("assistant", reply)
                logger.info(
                    "Intent add_event from %s — created event '%s' on %s at %s (id=%s)",
                    user_name, event_summary, date_str, time_str, event_id,
                )

            except Exception as exc:
                # Calendar API error, parsing error, etc. — don't expose
                # the raw exception to the user.
                logger.error(
                    "Failed to create calendar event for %s: %s",
                    user_name, exc, exc_info=True,
                )
                reply = (
                    "⚠️ Sorry, I couldn't create that event — "
                    "please try again in a moment."
                )
                await update.message.reply_text(reply)
                save_message("assistant", reply)

    else:
        # ---- CHAT intent (default fallback) ----
        # No actionable intent detected — use the full conversational
        # reply pipeline with task, calendar, and history context.

        # Fetch the upcoming week of calendar events so the LLM can
        # answer planning questions beyond just today.  We use
        # get_upcoming_events() here instead of get_today_events() —
        # /agenda still uses the today-only version for its quick view.
        try:
            today_events = get_upcoming_events(days_ahead=7)
        except Exception as exc:
            logger.warning("Could not fetch calendar for chat context: %s", exc)
            today_events = []

        reply = generate_reply(
            user_message=user_text,
            open_tasks=open_tasks,
            today_events=today_events,
            recent_messages=recent_messages,
        )

        await update.message.reply_text(reply)
        save_message("assistant", reply)
        logger.info("Chat with %s — user: %s | reply: %s", user_name, user_text[:80], reply[:80])

# ---------------------------------------------------------------------------
# 4. Build the application and start polling
# ---------------------------------------------------------------------------

def main() -> None:
    """Create the bot application, register handlers, and start polling."""
    logger.info("Starting Teena bot...")

    # Initialise the database — creates tables if they don't exist yet.
    # We do this ONCE at startup, before the bot begins processing messages.
    init_db()

    # Build the Application with our token
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register handlers — order matters:
    #   • CommandHandlers are checked first (in the order they're added)
    #   • The MessageHandler catch-all at the end handles everything else
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("addtask", addtask_command))
    app.add_handler(CommandHandler("tasks", tasks_command))
    app.add_handler(CommandHandler("done", done_command))
    app.add_handler(CommandHandler("agenda", agenda_command))  # Calendar agenda
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))  # Gemini chat

    # Start polling — the bot will keep running until you press Ctrl+C
    logger.info("Bot is polling for updates… Press Ctrl+C to stop.")
    app.run_polling()


# This guard ensures main() only runs when you execute the file directly
# (not when it's imported as a module by something else).
if __name__ == "__main__":
    main()
