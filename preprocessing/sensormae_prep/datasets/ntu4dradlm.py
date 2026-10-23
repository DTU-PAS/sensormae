from pathlib import Path

import cv2
import h5py
import numpy as np

from .. import rosbag
from ..common import crop_resize, fov_box, load_yaml, nearest, undistort
from .base import Dataset

CALIB = Path(__file__).resolve().parents[2] / "calib" / "ntu4dradlm.yaml"
SEQUENCES = ("garden", "loop1", "loop2", "loop3", "smoke_cp")  # cp and nyl have no thermal
MAX_DT_NS = 50_000_000


def jpeg(img):
    """JPEG encode/decode round trip (frames are stored as .jpg after extraction and crop)."""
    return cv2.imdecode(cv2.imencode(".jpg", img)[1], cv2.IMREAD_UNCHANGED)


class NTU4DRadLM(Dataset):
    """RGB and LWIR from the NTU4DRadLM rosbags, undistorted and matched within 50 ms
    (LWIR nearest to every selected RGB frame). The RGB is cropped to the LWIR FoV and
    both are resized to 640x512. Ids are {sequence index}_{RGB timestamp}, the sequence
    index following SEQUENCES."""

    name = "ntu4dradlm"
    manual = ("https://github.com/junzhang2016/NTU4DRadLM : download the rosbags of garden, "
              "loop1, loop2, loop3 and smoke_cp into ntu4dradlm/<sequence>/*.bag")
    required = SEQUENCES

    def __init__(self, root, work_dir):
        super().__init__(root, work_dir)
        c = self.c = load_yaml(CALIB)
        K = lambda k: np.array(c[k], np.float64).reshape(3, 3)  # noqa: E731
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = np.array(c["R_lwir2rgb"]), np.array(c["T_lwir2rgb"])
        self.size = (c["w_lwir"], c["h_lwir"])
        self.box_rgb = fov_box(K("K_lwir"), self.size, K("K_rgb"), (c["w_rgb"], c["h_rgb"]), T)

    def _undistort(self, img, cam):
        c = self.c
        return undistort(img, c[f"K_{cam}"], c[f"D_{cam}"], (c[f"w_{cam}"], c[f"h_{cam}"]))

    def prepare(self, ids):
        """Extract the undistorted frames of the given ids into work/<sequence>.h5."""
        wanted = {}
        for sample_id in ids:
            idx, name = sample_id.split("_", 1)
            wanted.setdefault(SEQUENCES[int(idx) - 1], set()).add(name)
        self.work.mkdir(parents=True, exist_ok=True)
        for seq, names in wanted.items():
            out = self.work / f"{seq}.h5"
            if out.exists():
                with h5py.File(out, "r") as f:
                    if names <= set(f):
                        continue
            self._extract(seq, names, out)

    def _extract(self, seq, names, out):
        c = self.c
        bags = sorted((self.root / seq).glob("*.bag"))
        topics = {"rgb": c["rgb_topic"], "lwir": c["lwir_topic"]}

        def messages(kind):
            for bag in bags:
                with rosbag.open_bag(bag) as r:
                    conns = [x for x in r.connections if x.topic == topics[kind]]
                    for _, _, raw in r.messages(conns):
                        s, ns = rosbag.stamp(raw)
                        yield f"{s:09d}_{ns:09d}", raw

        lwir_names = sorted(n for n, _ in messages("lwir"))
        lwir_ts = np.array([int(n.replace("_", "")) for n in lwir_names], np.int64)
        lwir_for = {}
        for name in names:
            i = nearest(lwir_ts, int(name.replace("_", "")), MAX_DT_NS)
            if i is not None:
                lwir_for.setdefault(lwir_names[i], []).append(name)

        with h5py.File(out, "w") as f:
            for kind in ("rgb", "lwir"):
                for msg_name, raw in messages(kind):
                    targets = [msg_name] if kind == "rgb" and msg_name in names else \
                        lwir_for.get(msg_name, []) if kind == "lwir" else []
                    if not targets:
                        continue
                    img = jpeg(self._undistort(rosbag.compressed_image(raw), kind))
                    for name in targets:
                        f.require_group(name).create_dataset(kind, data=img, compression="lzf")

    def load(self, split, sample_id):
        idx, name = sample_id.split("_", 1)
        with h5py.File(self.work / f"{SEQUENCES[int(idx) - 1]}.h5", "r") as f:
            raw = {k: v[()] for k, v in f[name].items()}
        rgb = jpeg(crop_resize(raw["rgb"], self.box_rgb, self.size))
        lwir = jpeg(crop_resize(raw["lwir"], (0, 0, *self.size), self.size))
        # Both were read back as 3-channel BGR images: RGB order, thermal keeps the R channel.
        return {"rgb": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                "lwir": cv2.cvtColor(lwir, cv2.COLOR_BGR2RGB)}
