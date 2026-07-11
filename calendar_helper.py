"""
calendar_helper.py — Google Calendar integration module.

Provides functions to authenticate with Google Calendar API
and fetch today's events from the user's primary calendar.

Usage:
    from calendar_helper import get_today_events
    events = get_today_events()
"""

import os
import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Read-only access to the user's calendar
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Paths are resolved relative to this file's directory (project root)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(_BASE_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(_BASE_DIR, "credentials.json")


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #

def get_calendar_service():
    """
    Build and return an authorized Google Calendar API service object.

    Authentication flow:
      1. If token.json exists and contains valid credentials, reuse them.
      2. If the token is expired but refreshable, refresh it automatically.
      3. Otherwise, run the full OAuth2 InstalledAppFlow (opens a browser
         window so the user can log in and authorize the app).
      4. Persist the resulting credentials to token.json for future runs.

    Raises:
        FileNotFoundError: If credentials.json is missing from the project root.

    Returns:
        googleapiclient.discovery.Resource: Authorized Calendar v3 service.
    """
    creds = None

    # Step 1 — Try to load existing credentials from token.json
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    # Step 2 — If no valid credentials are available, obtain new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Token expired but we can refresh it silently
            creds.refresh(Request())
        else:
            # No usable token — run the full OAuth consent flow
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing '{CREDENTIALS_PATH}'.\n"
                    "Please download your OAuth 2.0 Client ID JSON from the "
                    "Google Cloud Console (APIs & Services → Credentials) and "
                    "save it as 'credentials.json' in the project root."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            # This opens the default browser for the user to authorize
            creds = flow.run_local_server(port=0)

        # Step 3 — Save credentials so we don't need to re-authorize next time
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    # Step 4 — Build and return the Calendar API service
    service = build("calendar", "v3", credentials=creds)
    return service


# --------------------------------------------------------------------------- #
# Event Fetching
# --------------------------------------------------------------------------- #

def get_today_events():
    """
    Fetch today's events from the user's primary Google Calendar.

    The date range spans from midnight to midnight in the system's local
    timezone so the results match what the user sees on their own calendar.

    Returns:
        list[dict]: A list of event dicts sorted by start time, each with:
            - summary  (str): Event title.
            - start    (str): Formatted start time, e.g. "14:30" or "All day".
            - end      (str): Formatted end time,   e.g. "15:00" or "All day".
        Returns an empty list when there are no events.
    """
    service = get_calendar_service()

    # Determine today's date boundaries in the local timezone
    now = datetime.datetime.now().astimezone()          # current local time
    local_tz = now.tzinfo

    start_of_day = datetime.datetime.combine(
        now.date(), datetime.time.min, tzinfo=local_tz
    )
    end_of_day = datetime.datetime.combine(
        now.date(), datetime.time.max, tzinfo=local_tz
    )

    # Query the Calendar API — results are returned in ascending startTime
    events_result = service.events().list(
        calendarId="primary",
        timeMin=start_of_day.isoformat(),
        timeMax=end_of_day.isoformat(),
        singleEvents=True,              # expand recurring events
        orderBy="startTime",
    ).execute()

    raw_events = events_result.get("items", [])

    # Parse each event into a simplified dict
    parsed = []
    for event in raw_events:
        summary = event.get("summary", "(No title)")

        start_info = event.get("start", {})
        end_info = event.get("end", {})

        # All-day events use "date"; timed events use "dateTime"
        if "date" in start_info:
            start_str = "All day"
            end_str = "All day"
        else:
            start_str = _format_time(start_info.get("dateTime", ""))
            end_str = _format_time(end_info.get("dateTime", ""))

        parsed.append({
            "summary": summary,
            "start": start_str,
            "end": end_str,
        })

    return parsed


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _format_time(iso_string: str) -> str:
    """
    Convert an ISO-8601 datetime string to a short "H:MM AM/PM" time string
    in the system's local timezone.

    Args:
        iso_string: An ISO-8601 datetime string (e.g. "2026-07-10T14:30:00+05:30").

    Returns:
        A formatted time string like "2:30 PM".
    """
    if not iso_string:
        return ""
    dt = datetime.datetime.fromisoformat(iso_string).astimezone()
    return dt.strftime("%I:%M %p").lstrip("0")
