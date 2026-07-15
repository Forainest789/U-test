## ComfyUI workflow for SlotMem + Wan2.2
This optional workflow demonstrates the Wan2.2 video wrapper setup used alongside SlotMem experiments.

The workflow JSON files in this directory are based on [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).

Note: The cfg value in the original inference setting is typically 5. However, due to the acceleration components integrated within the ComfyUI workflow, the cfg value cannot be maintained consistently. Consequently, we have adjusted and fixed the cfg at 1.5. **For improved textual alignment, you may slightly increase the cfg value within the recommended range of 1 to 2.** 
