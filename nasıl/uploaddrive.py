import os
from pathlib import Path
from mimetypes import guess_type

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILE = "token.json"
CREDENTIALS_FILE = "credentials.json"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v", ".webm"}


def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def upload_one(service, file_path: str, parent_folder_id: str | None = None):
    mime_type, _ = guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    metadata = {"name": os.path.basename(file_path)}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    media = MediaFileUpload(
        file_path,
        mimetype=mime_type,
        resumable=True,
        chunksize=10 * 1024 * 1024,
    )

    request = service.files().create(
        body=metadata,
        media_body=media,
        fields="id,name",
        supportsAllDrives=True,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  {Path(file_path).name}: {int(status.progress() * 100)}%")
    return response


if __name__ == "__main__":
    service = get_drive_service()

    local_folder = os.path.join(os.getcwd(), "out_clips")
    drive_parent_folder_id = None  # or your Drive folder ID

    paths = [p for p in Path(local_folder).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    print(f"Found {len(paths)} videos in {local_folder}")

    for p in paths:
        print("Uploading:", p.name)
        upload_one(service, str(p), drive_parent_folder_id)

    print("Done.")
