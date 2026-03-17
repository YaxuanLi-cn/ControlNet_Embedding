import json
import os
import glob

# Paths
src_dir = '/root/dreamNav/pairUAV/train'
img_dir = '/root/dreamNav/pairUAV/tours'
out_dir = '/root/ControlNet/training/pairUAV'

# Create output directory
os.makedirs(out_dir, exist_ok=True)

json_files = sorted(glob.glob(os.path.join(src_dir, '**', '*.json'), recursive=True))

out_path = os.path.join(out_dir, 'prompt.json')
count = 0
with open(out_path, 'w') as f:
    for jf in json_files:
        with open(jf, 'r') as jfp:
            data = json.load(jfp)

        heading = data['heading_num']
        rng = data['range_num']

        h_dir = "right" if heading >= 0 else "left"
        r_dir = "forward" if rng >= 0 else "backward"

        prompt = (
            f"Aerial drone view after turning {h_dir} {abs(heading):.0f} degrees "
            f"and moving {r_dir} {abs(rng):.0f} meters"
        )

        entry = {
            "source": os.path.join(img_dir, data["image_a"]),
            "target": os.path.join(img_dir, data["image_b"]),
            "prompt": prompt,
        }
        f.write(json.dumps(entry) + '\n')
        count += 1

print(f"Done. Wrote {count} entries to {out_path}")