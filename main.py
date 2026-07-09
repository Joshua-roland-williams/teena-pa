"""
main.py — Teena, your Telegram personal assistant bot.

This is the entry point for the bot. It handles:
  • /start    → sends a welcome message
  • /addtask  → adds a new task to the database
  • /tasks    → lists all open (incomplete) tasks
  • /done     → marks a task as completed by its id
  • Any text  → echoes it back (placeholder for real AI logic later)

Uses python-telegram-bot v21.x (async style) with polling.
"""

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
from database import init_db, add_task, get_open_tasks, mark_task_done

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
        "• `/done <id>` — mark a task complete\n\n"
        "More is on the way — Calendar sync, mood tracking, and real conversation are coming soon!"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")
    logger.info("Sent welcome message to %s", user_first_name)


# ---------------------------------------------------------------------------
# Task management command handlers
# ---------------------------------------------------------------------------

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
    each with keys: id, text, due_date, done, created_at.
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
# Echo handler (catch-all for plain text)
# ---------------------------------------------------------------------------

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle any plain text message — placeholder until real chat (Phase 4) is built."""
    await update.message.reply_text(
        "I can't chat freely just yet — that's coming soon! 🙂\n"
        "For now, try:\n"
        "• `/addtask <text>`\n"
        "• `/tasks`\n"
        "• `/done <id>`",
        parse_mode="Markdown",
    )
    logger.info(
        "Received free-text from %s (chat not yet implemented): %s",
        update.effective_user.first_name,
        update.message.text,
    )

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
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start polling — the bot will keep running until you press Ctrl+C
    logger.info("Bot is polling for updates… Press Ctrl+C to stop.")
    app.run_polling()


# This guard ensures main() only runs when you execute the file directly
# (not when it's imported as a module by something else).
if __name__ == "__main__":
    main()
