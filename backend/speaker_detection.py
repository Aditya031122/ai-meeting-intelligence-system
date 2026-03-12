import re


def detect_speakers(transcript: str):
    """
    Detect speakers in transcript if format like:
    Alex: text
    Rahul: text
    """

    speaker_pattern = r"([A-Z][a-z]+):\s*(.*)"

    results = []

    for line in transcript.split("\n"):
        match = re.match(speaker_pattern, line.strip())
        if match:
            speaker = match.group(1)
            text = match.group(2)

            results.append({
                "speaker": speaker,
                "text": text
            })

    return results


if __name__ == "__main__":

    transcript = """
    Alex: We should launch the drone delivery system next week.
    Rahul: Backend deployment will be ready by Friday.
    Priya: I'll finalize the deployment checklist.
    """

    speakers = detect_speakers(transcript)

    print(speakers)