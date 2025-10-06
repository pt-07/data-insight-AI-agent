from src.ingestion import fetch_from_drive
from src.agent import generate_user_personas
import config

def main():
    #formatting 
    print("=" * 60)
    print("AI-POWERED USER PERSONA GENERATOR")
    print("=" * 60)
    #fetch data
    print("\nFetching data from Google Drive...")
    datasets = fetch_from_drive(folder_id=config.DRIVE_FOLDER_ID)
    print(f"\nLoaded {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"  • {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    #generate random user personas
    print("\n[Step 2] Generating AI-powered user personas...")
    num_users = 5
    personas, user_data = generate_user_personas(datasets, num_users=num_users)
    #display
    print("\n" + "=" * 60)
    print("GENERATED PERSONAS")
    print("=" * 60)
    print(personas)
    output_file = "user_personas_output.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AI-GENERATED USER PERSONAS\n")
        f.write("=" * 60 + "\n\n")
        f.write(personas)
    print(f"\n✓ Personas saved to: {output_file}")
    print("\n[Complete] Run again to generate new random personas!")

if __name__ == "__main__":
    main()