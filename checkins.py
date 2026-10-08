"""
checkins.py — Proactive morning and evening check-in handlers for Teena Bot.

This module contains the scheduled check-in handlers that run via PTB's
JobQueue (run_daily).  Each handler:
  1. Retrieves the persisted chat_id (no incoming Update in scheduled jobs).
  2. Gathers REAL data from the database and calendar — never invents anything.
  3. Calls generate_checkin_message() in llm_helper.py with a kind-specific
     instruction block.
  4. Sends the result to the user via context.bot.send_message().
  5. Wraps everything in try/except — never raises, only logs.

Registered in main.py with tz-aware run_daily calls.
"""

import asyncio
import datetime
import logging

from telegram.ext import ContextTypes

# Database helpers — import only what we need
from database import (
    APP_TZ, get_chat_id, get_open_tasks, get_tasks_completed_today,
    get_recent_moods, get_mood_average, save_message,
)

# Calendar helpers
from calendar_helper import get_today_events, get_upcoming_events

# LLM helper for generating check-in messages
from llm_helper import generate_checkin_message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Morning check-in
# ---------------------------------------------------------------------------

async def morning_checkin(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Proactive morning check-in — runs daily at 08:00 IST.

    Gathers today's calendar events, open tasks (prioritised by due date
    and priority level), and recent mood data, then calls
    generate_checkin_message("morning", ...) to produce a concise,
    honest morning briefing.

    If the chat_id has not been saved yet (user hasn't sent /start),
    logs a warning and returns silently.
    """
    chat_id = get_chat_id()

    if chat_id is None:
        logger.warning("morning_checkin: no chat_id saved yet — skipping.")
        return

    try:
        # Gather REAL data only — no assumptions or fabrications
        open_tasks = get_open_tasks()
        recent_moods = get_recent_moods(limit=7)
        mood_avg_7 = get_mood_average(days=7)

        # Today's calendar events
        try:
            today_events = await asyncio.to_thread(get_today_events)
        except Exception as cal_exc:
            logger.warning("morning_checkin: could not fetch calendar: %s", cal_exc)
            today_events = []

        # Generate the check-in message via the LLM (run in thread)
        message = await asyncio.to_thread(
            generate_checkin_message,
            "morning",
            open_tasks=open_tasks,
            today_events=today_events,
            recent_moods=recent_moods,
            mood_avg_7=mood_avg_7,
        )

        await context.bot.send_message(chat_id=chat_id, text=message)

        try:
            save_message("assistant", message)
        except Exception as save_exc:
            logger.warning("morning_checkin: could not save message: %s", save_exc)

        logger.info("morning_checkin: sent to chat_id=%d (%d chars)", chat_id, len(message))

    except Exception as exc:
        # NEVER raise — log and send a minimal fallback
        logger.error("morning_checkin failed: %s", exc, exc_info=True)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "🌅 Good morning! I had trouble putting your briefing "
                    "together — try /tasks and /agenda to see what's on "
                    "your plate today."
                ),
            )
        except Exception:
            logger.error("morning_checkin: even fallback send failed", exc_info=True)


# ---------------------------------------------------------------------------
# Evening check-in
# ---------------------------------------------------------------------------

async def evening_checkin(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Proactive evening check-in — runs daily at 21:00 IST.

    Gathers tasks completed today, remaining open tasks, tomorrow's
    calendar preview, and recent mood data, then calls
    generate_checkin_message("evening", ...) to produce a warm wrap-up
    that ends by asking the user to rate their day 1-10.

    If the chat_id has not been saved yet, logs a warning and returns.
    """
    chat_id = get_chat_id()

    if chat_id is None:
        logger.warning("evening_checkin: no chat_id saved yet — skipping.")
        return

    try:
        # Gather REAL data only
        completed_today = get_tasks_completed_today()
        open_tasks = get_open_tasks()
        recent_moods = get_recent_moods(limit=7)
        mood_avg_7 = get_mood_average(days=7)

        # Today's events (for context in the wrap-up)
        try:
            today_events = await asyncio.to_thread(get_today_events)
        except Exception as cal_exc:
            logger.warning("evening_checkin: could not fetch today's calendar: %s", cal_exc)
            today_events = []

        # Tomorrow's calendar preview
        try:
            upcoming = await asyncio.to_thread(get_upcoming_events, days_ahead=2)
            tomorrow = datetime.datetime.now(APP_TZ).date() + datetime.timedelta(days=1)
            tomorrow_events = [
                e for e in upcoming if e.get("date_obj") == tomorrow
            ]
        except Exception as cal_exc:
            logger.warning("evening_checkin: could not fetch tomorrow's calendar: %s", cal_exc)
            tomorrow_events = []

        # Generate the check-in message via the LLM (run in thread)
        message = await asyncio.to_thread(
            generate_checkin_message,
            "evening",
            open_tasks=open_tasks,
            completed_today=completed_today,
            today_events=today_events,
            tomorrow_events=tomorrow_events,
            recent_moods=recent_moods,
            mood_avg_7=mood_avg_7,
        )

        await context.bot.send_message(chat_id=chat_id, text=message)

        try:
            save_message("assistant", message)
        except Exception as save_exc:
            logger.warning("evening_checkin: could not save message: %s", save_exc)

        logger.info("evening_checkin: sent to chat_id=%d (%d chars)", chat_id, len(message))

    except Exception as exc:
        # NEVER raise — log and send a minimal fallback
        logger.error("evening_checkin failed: %s", exc, exc_info=True)
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "🌙 Good night! I had trouble putting the evening "
                    "summary together, but I hope you had a good day. "
                    "How would you rate today, 1 to 10? 💙"
                ),
            )
        except Exception:
            logger.error("evening_checkin: even fallback send failed", exc_info=True)
