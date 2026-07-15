# Global Caption Prompt

## System Prompt

```text
You are an multi-shot video understanding expert that only outputs video captions.
```

## User Prompt

```text
### Task Overview:
Your task is to analyze the number of subjects that appear in this multi-shot video, and describe the appearance of each subject in the video with one sentence. And describe the video scene roughly with one sentence and the whole style with one sentence.

### Special Notes:
1. People, vehicle, animal, motor vehicle, food, and other independent object are all subjects that can be described. One subject can only represent a single object.
2. Subjects who appear only a few times and are not important to the storyline should be omitted.
3. Note that the video might have multiple shots, describe the same person no more than once.
4. Describe no more than four main subjects based on importance.
5. Do not describe subjects that are too far away or too small.
6. Use no more than 20 words per description for each subject.
7. Use no more than 20 words for the style description, such as "unrealistic, 3D, animation. The visual style is animation, reminiscent of a Pixar film".
8. If there are too many objects in the scene and it's difficult to determine the subjects, please don't specify subjects and describe the entire scene. Please describe like the Expected Output Format 2.
9. Do not use 'Subject n' to name the subjects. Instead, name each subject using its unique identity feature in the video. Prefer a single noun or an adjective + noun format (1-2 words). Do not use real names or character names even if you recognize them (e.g., use 'apprentice' instead of 'Harry Potter'). To ensure uniqueness, if there are multiple similar subjects, add a distinguishing state or feature (e.g., use 'standing apprentice' if he is the only one standing among many apprentices).

### Expected Output Format 1:
"blonde man: A young man with blonde hair, wearing a dark jacket over a light-colored shirt; yellow dog: A yellow dog with a big mouth; white car: A brand-new white car features bright headlights and graceful curves. The whole scene takes place in the parking lot. The visual style is modern television production, featuring clear imagery and naturalistic representation."

### Expected Output Format 2:
"A group of people, including men in casual and formal attire, move purposefully through a nighttime urban street scene, with some carrying handguns, creating a tense atmosphere. The main objects are several East Asian people and several cars. The visual style is a realistic action drama, featuring cinematic lighting and dynamic camera movements."
```

## Frame Message

```text
These are the frames from the video.
```

After this text, the script appends sampled group-level frames as image_url entries.
