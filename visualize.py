import torch
import numpy as np
import napari

# Load the saved volume
volume = torch.load("volumes/OAS1_0001_MR1_mpr-1.pt") 
volume_np = volume.squeeze(0).numpy()

if volume_np.max() > 1:
    volume_np = volume_np / 255.0

viewer = napari.Viewer()
viewer.add_image(volume_np, name='MRI Volume', colormap='gray', rendering='mip')

napari.run()
