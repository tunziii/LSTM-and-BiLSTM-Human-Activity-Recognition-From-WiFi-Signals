import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csiread

input_root = r"D:\Experiment-2\realdata\input_data\30ms\validation"
output_root = r"D:\DatenExperiment2\validation"

os.makedirs(output_root, exist_ok=True)

# Process folder
for activity in os.listdir(input_root):

    activity_path = os.path.join(input_root, activity)

    if not os.path.isdir(activity_path):
        continue

    # Create matching output folder
    output_activity = os.path.join(output_root, activity)
    os.makedirs(output_activity, exist_ok=True)

    # Read .pcap
    for filename in os.listdir(activity_path):

        if not filename.lower().endswith(".pcap"):
            continue

        filepath = os.path.join(activity_path, filename)

        print(f"File: {filename}")

        # Read CSI
        csidata = csiread.Nexmon(
            filepath,
            chip="4366c0",
            bw=80
        )

        csidata.read()

        amplitude = np.abs(csidata.csi)  # Convert complex CSI to amplitude

        scaler = MinMaxScaler()  # Normalize
        amplitude = scaler.fit_transform(amplitude)

        columns = [f"SC_{i}" for i in range(256)]  # Create DataFrame
        df = pd.DataFrame(amplitude, columns=columns)

        df["Label"] = activity  # Add label

        # #Save as .csv
        output_file = os.path.join(
            output_activity,
            os.path.splitext(filename)[0] + ".csv"
        )

        df.to_csv(output_file, index=False)

        print(f"Saved in {output_file}")

print("\nDone")