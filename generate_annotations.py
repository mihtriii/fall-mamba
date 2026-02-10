import os
import glob
import pandas as pd

def generate_annotations(root_dir, output_file):
    data = []
    classes = {'ADL': 0, 'Fall': 1}
    
    print(f"Scanning {root_dir}...")
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        for class_name, label in classes.items():
            if class_name in dirs:
                class_dir = os.path.join(root, class_name)
                
                # Support common video extensions
                video_files = []
                for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
                    video_files.extend(glob.glob(os.path.join(class_dir, ext)))
                
                if video_files:
                    print(f"Found {len(video_files)} videos for class {class_name} in {class_dir}")
                    
                for video_path in video_files:
                    # Store absolute path
                    abs_path = os.path.abspath(video_path)
                    data.append([abs_path, label])
            
    df = pd.DataFrame(data, columns=['path', 'label'])
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(output_file, index=False, header=False, sep=' ')
    print(f"Saved {len(df)} annotations to {output_file}")

if __name__ == "__main__":
    base_dir = "Strategy1_Combined"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    generate_annotations(train_dir, "train.csv")
    generate_annotations(val_dir, "val.csv")
