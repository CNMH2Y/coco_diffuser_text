import numpy as np
import torch

from diffusers import DDIMScheduler, DDPMPipeline
from tqdm.auto import tqdm
import cv2


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # Create new scheduler and set num inference steps
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=100)

    # The random starting point
    x = torch.randn(1, 3, 256, 256).to(device)  # Batch of 4, 3-channel 256 x 256 px images

    # Loop through the sampling timesteps
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        # Prepare model input
        model_input = scheduler.scale_model_input(x, t)

        # Get the prediction
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]

        # Calculate what the updated sample should look like with the scheduler
        scheduler_output = scheduler.step(noise_pred, t, x)

        # Update x
        x = scheduler_output.prev_sample

        img = (x[0].permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5) * 256
        img = img.numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("", img)
        cv2.waitKey(100)

    print("Press any key to exit.")
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
