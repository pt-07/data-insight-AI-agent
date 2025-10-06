from src.ingestion import fetch_from_drive
from src.conversational_agent import start_conversation
import config

def main():
    print("Loading data from Google Drive...")
    datasets = fetch_from_drive(folder_id=config.DRIVE_FOLDER_ID)
    print(f"\n✓ Loaded {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"  • {name}: {df.shape[0]:,} rows")
    start_conversation(datasets)

if __name__ == "__main__":
    main()