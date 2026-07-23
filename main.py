"""
main.py — Teena, your Telegram personal assistant bot.

This is the entry point for the bot. It handles:
  • /start    → sends a welcome message
  • /addtask  → adds a new task to the database
  • /tasks    → lists all open (incomplete) tasks
  • /done     → marks a task as completed by its id
  • /agenda   → shows today's Google Calendar events
  • /mood     → logs a mood score (1-10) with an optional note
  • Any text  → Gemini-powered intent detection first (can add tasks,
                 complete tasks, schedule calendar events, reschedule
                 existing events, or log moods via natural language),
                 then falls back to conversational chat.

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
from database import init_db, add_task, get_open_tasks, mark_task_done, delete_task, get_completed_tasks, save_message, get_recent_messages, log_mood, get_recent_moods, get_mood_average

# Import the Google Calendar helper for the /agenda command, event creation,
# event rescheduling, and upcoming-week context for chat replies
from calendar_helper import get_today_events, get_upcoming_events, create_event, update_event

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
        "• `/agenda` — see today's calendar events\n"
        "• `/mood <1-10> [note]` — log how you're feeling\n\n"
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
# Mood tracking command handler
# ---------------------------------------------------------------------------

async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /mood <score 1-10> [optional note] — log a mood entry.

    The first argument must be an integer 1-10.  Everything after it is
    treated as an optional free-text note.
    """

    # If the user just types "/mood" with nothing after it, show usage.
    if not context.args:
        await update.message.reply_text(
            "Usage: `/mood <score 1-10> [note]`\n"
            "Example: `/mood 7 feeling pretty good today`",
            parse_mode="Markdown",
        )
        return

    # Parse the score — must be an integer 1-10
    try:
        score = int(context.args[0])
    except ValueError:
        await update.message.reply_text(
            f"`{context.args[0]}` isn't a valid score.\n"
            "Please use a number from 1 to 10, e.g. `/mood 7`",
            parse_mode="Markdown",
        )
        return

    if not 1 <= score <= 10:
        await update.message.reply_text(
            "Score should be between 1 and 10.\n"
            "Example: `/mood 7 feeling pretty good today`",
            parse_mode="Markdown",
        )
        return

    # Everything after the score is the optional note
    note = " ".join(context.args[1:]) if len(context.args) > 1 else None

    # Persist the mood entry
    log_mood(score, note)

    # Reply with a brief, warm acknowledgment — vary tone based on score
    if score <= 3:
        # Low score: gentle, no advice, just acknowledgment
        reply = "Noted 💙 Thanks for sharing — hope the rest of the day is kinder to you."
    elif score <= 5:
        # Mid-low: simple acknowledgment
        reply = "Got it — logged. Hang in there 🤍"
    elif score <= 7:
        # Decent: brief positive
        reply = "Logged! Sounds like a solid day 👍"
    else:
        # High score: short celebratory
        reply = "Love that — logged! 🌟"

    await update.message.reply_text(reply)
    logger.info(
        "User %s logged mood: score=%d note=%s",
        update.effective_user.first_name, score, note,
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
         • add_task          → create the task in the DB, reply with confirmation.
         • complete_task      → validate & mark done, reply with confirmation.
         • delete_task        → validate & soft-delete, reply with confirmation.
         • add_event          → parse date/time, create a Google Calendar event.
         • reschedule_event   → validate event id, update via Calendar API.
         • log_mood           → log mood score + note, reply naturally.
         • chat (default)     → call generate_reply() for a conversational answer.
      5. Save the assistant's reply to the database.
    """
    user_text = update.message.text
    user_name = update.effective_user.first_name

    # Step 1 — Persist the incoming user message
    save_message("user", user_text)

    # Step 2 — Gather context for the LLM
    open_tasks = get_open_tasks()
    recent_messages = get_recent_messages(limit=10)

    # Fetch upcoming events ONCE and reuse for both intent detection and
    # the chat fallback — avoids a redundant second Calendar API call.
    try:
        upcoming_events = get_upcoming_events(days_ahead=7)
    except Exception as exc:
        logger.warning("Could not fetch upcoming events for intent context: %s", exc)
        upcoming_events = []

    # Step 3 — Detect the user's intent before falling back to normal chat.
    # detect_intent() always returns a dict with at least {"intent": "chat"}
    # so it's safe to branch on without extra null checks.
    # We pass recent_messages so the model can combine multi-turn requests
    # (e.g. user says "schedule a meeting", Teena asks "what time?", user
    # replies "3pm" → the model stitches these into one add_event intent).
    intent_data = detect_intent(user_text, open_tasks, upcoming_events, recent_messages)
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

    elif intent == "delete_task":
        # ---- DELETE TASK intent (soft-delete) ----
        # The user wants to REMOVE a task from the list — NOT complete it.
        # This is a soft-delete: the task stays in the database (for
        # history / potential undo) but is hidden from active views.
        #
        # We follow the exact same safety pattern as complete_task:
        # validate the Gemini-returned task_id against our open_tasks
        # list before doing anything, to guard against hallucinated or
        # stale ids.
        target_id = intent_data.get("task_id")

        # Build a set of valid open task ids for fast lookup
        open_ids = {t["id"] for t in open_tasks}

        if target_id is not None and target_id in open_ids:
            # ID is valid — soft-delete it
            delete_task(target_id)

            # Find the task text so we can confirm by name, not just id
            task_name = next(
                (t["text"] for t in open_tasks if t["id"] == target_id),
                f"task #{target_id}",
            )
            reply = f"🗑️ Removed *{task_name}* from your task list."
            await update.message.reply_text(reply, parse_mode="Markdown")
            save_message("assistant", reply)
            logger.info(
                "Intent delete_task from %s — soft-deleted task #%d: %s",
                user_name, target_id, task_name,
            )
        else:
            # ID is missing, invalid, or not in the open tasks list.
            # Don't silently delete the wrong task — show the list
            # so the user can clarify.
            lines = ["🤔 I wasn't sure which task you meant. Here are your open tasks:\n"]
            if open_tasks:
                for t in open_tasks:
                    due = f"  _(due {t['due_date']})_" if t.get("due_date") else ""
                    lines.append(f"  • [`{t['id']}`] {t['text']}{due}")
                lines.append("\nYou can say which one to delete, or tell me more.")
            else:
                lines.append("  (You have no open tasks right now.)")

            reply = "\n".join(lines)
            await update.message.reply_text(reply, parse_mode="Markdown")
            save_message("assistant", reply)
            logger.info(
                "Intent delete_task from %s — could not match task_id=%s",
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
                today_events=upcoming_events,
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

    elif intent == "reschedule_event":
        # ---- RESCHEDULE EVENT intent ----
        # Gemini matched the user's message to an existing calendar event
        # and returned the event_id plus any new_summary / new_date /
        # new_start_time fields.
        #
        # VALIDATE-BEFORE-ACTING pattern (same principle as tasks):
        # The LLM might hallucinate an event_id that doesn't actually
        # exist in the upcoming_events list we fetched.  We check it
        # against the real list before calling the Calendar API.
        target_event_id = intent_data.get("event_id")

        # Build a lookup of valid event ids from the list we fetched earlier
        valid_event_ids = {e["id"] for e in upcoming_events if e.get("id")}

        if target_event_id and target_event_id in valid_event_ids:
            # --- ID is valid — build the update parameters ---
            new_summary = intent_data.get("new_summary")  # str or None
            new_date_str = intent_data.get("new_date")      # "YYYY-MM-DD" or None
            new_time_str = intent_data.get("new_start_time")  # "HH:MM" or None

            # Find the original event dict so we can preserve its duration
            original_event = next(
                (e for e in upcoming_events if e["id"] == target_event_id), None
            )

            try:
                new_start_dt = None
                new_end_dt = None

                # If the user provided a new date and/or time, build
                # timezone-aware datetimes.  We preserve the original
                # event's duration when only the start changes.
                if new_date_str or new_time_str:
                    local_tz = datetime.datetime.now().astimezone().tzinfo

                    # Determine the original event's start/end so we can
                    # compute its duration and use defaults for missing fields.
                    orig_start_str = original_event.get("start", "") if original_event else ""
                    orig_end_str = original_event.get("end", "") if original_event else ""
                    orig_date_label = original_event.get("date", "") if original_event else ""

                    # Try to parse the original start/end times ("H:MM AM/PM")
                    # to compute the original duration. Default to 1 hour.
                    original_duration = datetime.timedelta(hours=1)
                    try:
                        if orig_start_str and orig_end_str and orig_start_str != "All day":
                            orig_start_time = datetime.datetime.strptime(
                                orig_start_str, "%I:%M %p"
                            ).time()
                            orig_end_time = datetime.datetime.strptime(
                                orig_end_str, "%I:%M %p"
                            ).time()
                            # Combine with a dummy date to compute delta
                            dummy_date = datetime.date.today()
                            orig_s = datetime.datetime.combine(dummy_date, orig_start_time)
                            orig_e = datetime.datetime.combine(dummy_date, orig_end_time)
                            if orig_e > orig_s:
                                original_duration = orig_e - orig_s
                    except (ValueError, TypeError):
                        pass  # Fall back to 1-hour default

                    # Resolve the new start date — use new_date_str if
                    # provided, otherwise keep today's date as a fallback.
                    if new_date_str:
                        start_date = datetime.date.fromisoformat(new_date_str)
                    else:
                        # No new date — keep the original event's date.
                        # orig_date_label is like "Mon, Jul 13" — parse it
                        # relative to the current year.
                        try:
                            parsed = datetime.datetime.strptime(
                                f"{orig_date_label} {datetime.date.today().year}",
                                "%a, %b %d %Y",
                            )
                            start_date = parsed.date()
                        except (ValueError, TypeError):
                            start_date = datetime.date.today()

                    # Resolve the new start time — use new_time_str if
                    # provided, otherwise keep the original event's time.
                    if new_time_str:
                        start_time = datetime.datetime.strptime(
                            new_time_str, "%H:%M"
                        ).time()
                    else:
                        # No new time — keep the original event's start time
                        try:
                            start_time = datetime.datetime.strptime(
                                orig_start_str, "%I:%M %p"
                            ).time()
                        except (ValueError, TypeError):
                            start_time = datetime.time(9, 0)  # safe fallback

                    naive_start = datetime.datetime.combine(start_date, start_time)
                    new_start_dt = naive_start.replace(tzinfo=local_tz)

                    # Preserve the original duration
                    new_end_dt = new_start_dt + original_duration

                # Call the Calendar API to apply the update
                updated_id = update_event(
                    event_id=target_event_id,
                    summary=new_summary,
                    start_datetime=new_start_dt,
                    end_datetime=new_end_dt,
                )

                # Build a human-friendly confirmation message
                confirm_parts = []
                if new_summary:
                    confirm_parts.append(f"title → *{new_summary}*")
                if new_start_dt:
                    friendly_time = new_start_dt.strftime("%I:%M %p").lstrip("0")
                    friendly_date = new_start_dt.strftime("%A, %B %d")
                    confirm_parts.append(f"time → {friendly_date} at {friendly_time}")

                event_name = new_summary or (
                    original_event.get("summary", "event") if original_event else "event"
                )
                changes_str = ", ".join(confirm_parts) if confirm_parts else "updated"
                reply = f"✏️ Rescheduled *{event_name}*: {changes_str}"
                await update.message.reply_text(reply, parse_mode="Markdown")
                save_message("assistant", reply)
                logger.info(
                    "Intent reschedule_event from %s — updated event '%s' (id=%s)",
                    user_name, event_name, updated_id,
                )

            except Exception as exc:
                # Calendar API error, date parsing error, etc.
                logger.error(
                    "Failed to reschedule event for %s: %s",
                    user_name, exc, exc_info=True,
                )
                reply = (
                    "⚠️ Sorry, I couldn't reschedule that event — "
                    "please try again in a moment."
                )
                await update.message.reply_text(reply)
                save_message("assistant", reply)
        else:
            # Event ID is missing, invalid, or not in the upcoming events list.
            # Don't guess — show the list so the user can clarify.
            lines = ["🤔 I wasn't sure which event you meant. Here are your upcoming events:\n"]
            if upcoming_events:
                for e in upcoming_events:
                    date_label = e.get("date", "")
                    if e["start"] == "All day":
                        lines.append(f"  • {date_label}: {e['summary']} _(all day)_")
                    else:
                        lines.append(f"  • {date_label}: {e['start']} – {e['end']}  {e['summary']}")
                lines.append("\nCan you tell me which one to reschedule?")
            else:
                lines.append("  (No upcoming events found.)")

            reply = "\n".join(lines)
            await update.message.reply_text(reply, parse_mode="Markdown")
            save_message("assistant", reply)
            logger.info(
                "Intent reschedule_event from %s — could not match event_id=%s",
                user_name, target_event_id,
            )

    elif intent == "log_mood":
        # ---- LOG MOOD intent (natural-language mood detection) ----
        # CONSERVATIVE CLASSIFICATION: This branch only fires when the
        # user genuinely shared how they're feeling (e.g. "feeling pretty
        # drained today").  detect_intent() is instructed to be conservative
        # — when in doubt it falls back to chat, so we don't put words in
        # the user's mouth or log a mood they didn't actually express.
        score = intent_data.get("score")
        note = intent_data.get("note")

        # Validate the score — if Gemini returned something unexpected,
        # just fall through to chat rather than crashing or logging junk.
        if isinstance(score, int) and 1 <= score <= 10:
            log_mood(score, note)

            # Brief, natural acknowledgment — not clinical, not therapy-speak.
            # Vary tone slightly based on the score range.
            if score <= 3:
                reply = "Heard you 💙 Thanks for sharing that."
            elif score <= 5:
                reply = "Noted — thanks for telling me 🤍"
            elif score <= 7:
                reply = "Gotcha 👍 Thanks for checking in."
            else:
                reply = "That's great to hear 🌟"

            await update.message.reply_text(reply)
            save_message("assistant", reply)
            logger.info(
                "Intent log_mood from %s — score=%d note=%s",
                user_name, score, note,
            )
        else:
            # Invalid score from the model — treat as a normal chat message
            logger.warning(
                "Intent log_mood from %s had invalid score=%s, falling back to chat",
                user_name, score,
            )
            completed_tasks = get_completed_tasks(limit=10)
            reply = generate_reply(
                user_message=user_text,
                open_tasks=open_tasks,
                today_events=upcoming_events,
                recent_messages=recent_messages,
                completed_tasks=completed_tasks,
            )
            await update.message.reply_text(reply)
            save_message("assistant", reply)

    else:
        # ---- CHAT intent (default fallback) ----
        # No actionable intent detected — use the full conversational
        # reply pipeline with task, calendar, history, and mood context.
        #
        # Reuse the upcoming_events we already fetched at the top of
        # chat() — no need to call get_upcoming_events() again.

        # Fetch recently completed tasks so the LLM can answer "what did
        # I finish?" using real data instead of guessing from conversation
        # history.  This closes the "what did I complete?" honesty gap.
        completed_tasks = get_completed_tasks(limit=10)

        # Fetch recent mood entries so Teena can be aware of how the user
        # is doing.  This feeds into the system prompt's RECENT MOOD
        # section — Teena will reference it sparingly and only when
        # genuinely relevant (see mood guidelines in _build_system_prompt).
        recent_moods = get_recent_moods(limit=7)

        reply = generate_reply(
            user_message=user_text,
            open_tasks=open_tasks,
            today_events=upcoming_events,
            recent_messages=recent_messages,
            completed_tasks=completed_tasks,
            recent_moods=recent_moods,
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
    app.add_handler(CommandHandler("mood", mood_command))       # Mood tracking
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))  # Gemini chat

    # Start polling — the bot will keep running until you press Ctrl+C
    logger.info("Bot is polling for updates… Press Ctrl+C to stop.")
    app.run_polling()


# This guard ensures main() only runs when you execute the file directly
# (not when it's imported as a module by something else).
if __name__ == "__main__":
    main()
