import os
import io
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Scopes required for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate():
    """Authenticate with Google Drive API using OAuth 2.0"""
    creds = None
    # Check if token.json exists (saved credentials)
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save credentials
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_drive_service():
    """Create and return Google Drive service"""
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    return service

def list_files_in_folder(folder_id):
    """List all files in a Google Drive folder"""
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size)"
    ).execute()
    files = results.get('files', [])
    return files

def download_file(file_id, file_name):
    """Download a file from Google Drive and return as pandas DataFrame"""
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_name}: {int(status.progress() * 100)}%")
    file_buffer.seek(0)
    # Read based on file type
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_buffer)
    elif file_name.endswith('.xlsx'):
        df = pd.read_excel(file_buffer)
    elif file_name.endswith('.json'):
        df = pd.read_json(file_buffer)
    else:
        raise ValueError(f"Unsupported file type: {file_name}")
    
    print(f"Loaded {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def fetch_from_drive(file_id=None, folder_id=None):
    """
    Fetch data from Google Drive
    - If file_id provided: download single file
    - If folder_id provided: download all files in folder
    """
    if file_id:
        # Get file metadata
        service = get_drive_service()
        file_metadata = service.files().get(fileId=file_id, fields='name').execute()
        file_name = file_metadata['name']
        return download_file(file_id, file_name)
    elif folder_id:
        # Download all files in folder
        files = list_files_in_folder(folder_id)
        datasets = {}
        for file in files:
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                continue  # Skip subfolders
            df = download_file(file['id'], file['name'])
            # Use filename without extension as key
            key = os.path.splitext(file['name'])[0]
            datasets[key] = df
        return datasets
    
    else:
        raise ValueError("Must provide either file_id or folder_id")

# Example usage
if __name__ == "__main__":
    # Test with your folder ID
    FOLDER_ID = "folder_id_here"
    print(f"Fetching files from folder ID: {FOLDER_ID}")
    # List files
    files = list_files_in_folder(FOLDER_ID)
    print(f"\nFound {len(files)} files:")
    for f in files:
        print(f"  - {f['name']} ({f['mimeType']})")
    
    # Download all files
    datasets = fetch_from_drive(folder_id=FOLDER_ID)
    print(f"\nLoaded {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")