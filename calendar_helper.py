"""
calendar_helper.py — Google Calendar integration module.

Provides functions to authenticate with Google Calendar API,
fetch today's events from the user's primary calendar, and
create new events.

Usage:
    from calendar_helper import get_today_events, create_event

    events   = get_today_events()
    event_id = create_event(summary, start_datetime, end_datetime)
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

# Full read/write access to the user's calendar (needed for creating events).
# NOTE: If you previously used "calendar.readonly", you MUST delete the
# existing token.json and re-authorize.  Google requires fresh consent
# whenever the requested permission level changes.
SCOPES = ["https://www.googleapis.com/auth/calendar"]

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


def get_upcoming_events(days_ahead: int = 7) -> list[dict]:
    """
    Fetch events from now until `days_ahead` days in the future.

    Unlike get_today_events() (which is scoped to a single day for the
    /agenda quick-view), this function gives the LLM a broader planning
    horizon so it can answer questions like "what does my week look like?"

    Each returned dict includes a ``date`` key (e.g. "Mon, Jul 13") so
    events from different days can be distinguished when displayed together.

    Parameters
    ----------
    days_ahead : int
        Number of days into the future to fetch (default 7, i.e. one week).

    Returns
    -------
    list of dict
        Each dict has keys: summary, start, end, date.
        Results are ordered by start time across all days.
        Returns an empty list when there are no upcoming events.
    """
    service = get_calendar_service()

    now = datetime.datetime.now().astimezone()           # current local time
    local_tz = now.tzinfo

    # End boundary: midnight at the end of the last day in the window
    end_date = now.date() + datetime.timedelta(days=days_ahead)
    end_boundary = datetime.datetime.combine(
        end_date, datetime.time.max, tzinfo=local_tz
    )

    # Query the Calendar API — from *now* (not midnight) through the window
    events_result = service.events().list(
        calendarId="primary",
        timeMin=now.isoformat(),
        timeMax=end_boundary.isoformat(),
        singleEvents=True,              # expand recurring events
        orderBy="startTime",
    ).execute()

    raw_events = events_result.get("items", [])

    parsed = []
    for event in raw_events:
        summary = event.get("summary", "(No title)")

        start_info = event.get("start", {})
        end_info = event.get("end", {})

        # All-day events use "date"; timed events use "dateTime"
        if "date" in start_info:
            start_str = "All day"
            end_str = "All day"
            # All-day events store the date as a plain "YYYY-MM-DD" string
            event_date = datetime.date.fromisoformat(start_info["date"])
        else:
            start_str = _format_time(start_info.get("dateTime", ""))
            end_str = _format_time(end_info.get("dateTime", ""))
            # Extract the date portion from the ISO datetime
            event_date = datetime.datetime.fromisoformat(
                start_info["dateTime"]
            ).astimezone().date()

        # Human-friendly date label, e.g. "Mon, Jul 13"
        date_label = event_date.strftime("%a, %b %d")

        parsed.append({
            "summary": summary,
            "start": start_str,
            "end": end_str,
            "date": date_label,
        })

    return parsed


# --------------------------------------------------------------------------- #
# Event Creation
# --------------------------------------------------------------------------- #

def create_event(
    summary: str,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
) -> str:
    """
    Insert a new event into the user's primary Google Calendar.

    Parameters
    ----------
    summary : str
        The event title / description.
    start_datetime : datetime.datetime
        Timezone-aware start time for the event.
    end_datetime : datetime.datetime
        Timezone-aware end time for the event.
        Must be after start_datetime.

    Returns
    -------
    str
        The Google Calendar event ID of the newly created event.

    Raises
    ------
    ValueError
        If the datetimes are not timezone-aware.
    RuntimeError
        If the Google Calendar API call fails for any reason.
    """
    # Ensure the caller passed timezone-aware datetimes — the API needs
    # an explicit timezone offset in the ISO string.
    if start_datetime.tzinfo is None or end_datetime.tzinfo is None:
        raise ValueError(
            "start_datetime and end_datetime must be timezone-aware "
            "(have tzinfo set).  Use .astimezone() or pass tz= to the "
            "datetime constructor."
        )

    # Build the event body per the Calendar API v3 spec.
    # The isoformat() output already includes the UTC offset (e.g.
    # "2026-07-12T14:00:00+05:30"), so a separate timeZone field is
    # unnecessary and could cause errors with non-IANA zone names.
    event_body = {
        "summary": summary,
        "start": {
            "dateTime": start_datetime.isoformat(),
        },
        "end": {
            "dateTime": end_datetime.isoformat(),
        },
    }

    try:
        service = get_calendar_service()
        created_event = (
            service.events()
            .insert(calendarId="primary", body=event_body)
            .execute()
        )
        return created_event["id"]

    except Exception as exc:
        raise RuntimeError(
            f"Failed to create calendar event '{summary}': {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _format_time(iso_string: str) -> str:
    """
    Convert an ISO-8601 datetime string to a short "H:MM AM/PM" time string
    in the system's local timezone.

    Parameters
    ----------
    iso_string : str
        An ISO-8601 datetime string (e.g. "2026-07-10T14:30:00+05:30").

    Returns
    -------
    str
        A formatted time string like "2:30 PM".
    """
    if not iso_string:
        return ""
    dt = datetime.datetime.fromisoformat(iso_string).astimezone()
    return dt.strftime("%I:%M %p").lstrip("0")
