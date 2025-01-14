from PIL import Image, ImageSequence
from tqdm import tqdm
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
from utils.constant import Constant
# Paths to your GIFs
C = Constant()
gif_paths = ["mgn.gif", "mp.gif","spatial.gif"]
gif_paths = [C.data_dir+"meshgrid/node_state_time_series_" + path for path in gif_paths]
# Load the GIFs
gifs = [Image.open(path) for path in gif_paths]

# Validate frame count and dimensions
frame_count = gifs[0].n_frames
width, height = gifs[0].size

if not all(gif.n_frames == frame_count for gif in gifs):
    raise ValueError("All GIFs must have the same number of frames.")
if not all(gif.size == (width, height) for gif in gifs):
    raise ValueError("All GIFs must have the same dimensions.")

cut = 210
new_width = 2 * width - 2 * cut
new_height = height

frames = []

# Process each frame
for frame_idx in tqdm(range(frame_count)):
    # Create a blank image for the current frame
    new_frame = Image.new("RGBA", (new_width, new_height))

    # Extract frames
    gifs[0].seek(frame_idx)
    full_frame = gifs[0].copy().crop((0,0,width-cut, height))  # Copy the current frame from gif1

    gifs[1].seek(frame_idx)
    right_half_2 = gifs[1].copy().crop((width // 2, 0, width-cut, height))

    gifs[2].seek(frame_idx)
    right_half_3 = gifs[2].copy().crop((width // 2, 0, width, height))

    # Paste the frames onto the canvas
    new_frame.paste(full_frame, (0, 0))  # gif1 on the left
    new_frame.paste(right_half_2, (width-cut, 0))  # gif2's right part
    new_frame.paste(right_half_3, (3 * width//2  - 2 * cut, 0))  # gif3's right part

    frames.append(new_frame)

# Save as a new GIF
frames[0].save(
    C.data_dir+"meshgrid/summary.gif",
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=gifs[0].info['duration'],  # Use the duration from gif1
)
