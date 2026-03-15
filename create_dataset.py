import os
import glob
import pandas as pd

def create_dataset():
    # 55 Indian Student Names
    NAMES = [
        "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna", "Ishaan", "Shaurya",
        "Atharv", "Advik", "Pranav", "Reyansh", "Moksh", "Dhruv", "Kabir", "Ritvik", "Aarush", "Kian",
        "Darsh", "Veer", "Ansh", "Laksh", "Dev", "Ishan", "Yash", "Samar", "Karaan", "Rudra",
        "Ananya", "Diya", "Saanvi", "Aditi", "Ira", "Kavya", "Ahana", "Riya", "Myra", "Navya",
        "Sara", "Meera", "Aarohi", "Priya", "Neha", "Pooja", "Anjali", "Shruti", "Swati", "Nidhi",
        "Roshni", "Simran", "Tanvi", "Vidya", "Kriti"
    ]
    id_to_name = {i: name for i, name in enumerate(NAMES, start=1)}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    proc_forg_dir = os.path.join(base_dir, 'processed_data', 'full_forg')
    proc_org_dir = os.path.join(base_dir, 'processed_data', 'full_org')

    data = []

    # Original = Label 0
    org_images = glob.glob(os.path.join(proc_org_dir, '*.*'))
    for img_path in org_images:
        filename = os.path.basename(img_path)
        # Format usually original_ID_NUM.png
        # Split by '_' to extract ID
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            student_id = int(parts[1])
            student_name = id_to_name.get(student_id, "Unknown")
            data.append({
                'File_Path': img_path,
                'ID': student_id,
                'Student_Name': student_name,
                'Label': 0
            })

    # Forgeries = Label 1
    forg_images = glob.glob(os.path.join(proc_forg_dir, '*.*'))
    for img_path in forg_images:
        filename = os.path.basename(img_path)
        # Format usually forgeries_ID_NUM.png
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            student_id = int(parts[1])
            student_name = id_to_name.get(student_id, "Unknown")
            data.append({
                'File_Path': img_path,
                'ID': student_id,
                'Student_Name': student_name,
                'Label': 1
            })

    df = pd.DataFrame(data)
    csv_path = os.path.join(base_dir, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Created dataset with {len(df)} records. Saved to {csv_path}")

if __name__ == "__main__":
    create_dataset()
