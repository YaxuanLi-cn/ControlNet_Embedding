import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset_path, img_dir):
        self.data_path = []
        self.img_dir = img_dir

        for tour_id in os.listdir(dataset_path):
            tour_dir = dataset_path + '/' + tour_id + '/'
            for json_name in os.listdir(tour_dir):
                self.data_path.append(tour_dir + json_name)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        
        now_json_path = self.data_path[idx]
        data = json.load(open(now_json_path))   

        heading = data['heading_num']
        rng = data['range_num']

        h_dir = "right" if heading >= 0 else "left"
        r_dir = "forward" if rng >= 0 else "backward"

        prompt = (
            f"Aerial drone view after turning {h_dir} {abs(heading):.0f} degrees "
            f"and moving {r_dir} {abs(rng):.0f} meters"
        )

        source_filename = os.path.join(self.img_dir, data["image_a"])
        target_filename = os.path.join(self.img_dir, data["image_b"])

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source,
                    heading_num=np.float32(heading),
                    range_num=np.float32(rng))

