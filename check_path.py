import os

# The exact path you are using in your other script
path_to_check = r"F:\dataset\inner-hostel-night\obj_train_data"

print(f"--- Checking path: {path_to_check} ---")

# 1. Check if the path exists at all
if os.path.exists(path_to_check):
    print("\n[SUCCESS] The folder path exists!")
    
    # 2. Get a list of everything inside the folder
    try:
        contents = os.listdir(path_to_check)
        
        # 3. Check if the folder is empty
        if not contents:
            print("\n[INFO] The folder exists but is empty. Make sure your image files are in it.")
        else:
            print(f"\n[INFO] Found {len(contents)} items in the folder. Here are the first 10:")
            # 4. Print the first 10 filenames
            for filename in contents[:10]:
                print(f"  - {filename}")
                
    except Exception as e:
        print(f"\n[ERROR] Could not read the folder's contents. Error: {e}")

else:
    print("\n[ERROR] The path does not exist! Please double-check for any typos in the path.")

print("\n--- Check complete ---")