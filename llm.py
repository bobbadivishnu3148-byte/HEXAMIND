import os
import re
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY not found. "
        "Check your .env file."
    )


# =========================================================
# OPENROUTER CLIENT
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


# =========================================================
# CLEAN EXPLANATION
# =========================================================

def clean_explanation(text):

    if not text:
        return (
            "The model predicted an output successfully, "
            "but the AI explanation was not returned."
        )

    text = str(text).strip()

    # Remove markdown fences
    text = text.replace("```text", "")
    text = text.replace("```", "")

    # -----------------------------------------------------
    # Find the REAL answer
    # -----------------------------------------------------

    match = re.search(
        r"(The model predicted\b.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return (
            "The model predicted the result based on "
            "patterns learned from the provided data. "
            "The probability values represent the model's "
            "estimated likelihood for each class, while "
            "confidence is not absolute certainty."
        )

    text = match.group(1).strip()

    # -----------------------------------------------------
    # Cut off prompt leakage appearing AFTER the answer
    # -----------------------------------------------------

    stop_markers = [
        "\nWe need to",
        "\nWe must",
        "\nLet's craft",
        "\nParagraph 1:",
        "\nParagraph 2:",
        "\nThe instruction",
        "\nRequired Elements",
        "\nConstraints",
        "\nAnalyze the Request",
        "\nRole:",
        "\nInput Data:",
        "\nImportant:",
        "\nYour task is",
        "\nWe also need",
        "\nWe should",
        "\nProbably include",
    ]

    earliest_position = len(text)

    for marker in stop_markers:

        position = text.find(marker)

        if position != -1 and position < earliest_position:
            earliest_position = position

    text = text[:earliest_position].strip()


    # -----------------------------------------------------
    # Remove accidental heading/bullet formatting
    # -----------------------------------------------------

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove numbered paragraph labels
        line = re.sub(
            r"^Paragraph\s+\d+\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE
        )

        # Remove bullet markers
        line = re.sub(
            r"^[-•*]\s*",
            "",
            line
        )

        cleaned_lines.append(line)


    text = " ".join(
        cleaned_lines
    ).strip()


    # -----------------------------------------------------
    # Guarantee correct beginning
    # -----------------------------------------------------

    if not text.lower().startswith(
        "the model predicted"
    ):

        return (
            "The model predicted the result based on "
            "patterns learned from the provided data. "
            "The probability values represent the model's "
            "estimated likelihood for each class, while "
            "confidence is not absolute certainty."
        )


    # -----------------------------------------------------
    # Limit excessive output
    # -----------------------------------------------------

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    if len(paragraphs) > 2:

        text = (
            paragraphs[0].strip()
            + "\n\n"
            + paragraphs[1].strip()
        )


    return text


# =========================================================
# PREDICTION EXPLANATION
# =========================================================

def explain_prediction(
    prediction,
    confidence,
    malignant_probability,
    benign_probability,
    top_features
):

    # -----------------------------------------------------
    # Feature information
    # -----------------------------------------------------

    feature_lines = []

    for item in top_features:

        feature_name = item.get(
            "feature",
            "Unknown feature"
        )

        contribution = float(
            item.get(
                "contribution",
                0
            )
        )

        direction = item.get(
            "direction",
            "unknown"
        )

        feature_lines.append(
            f"{feature_name}: "
            f"{contribution:+.6f} "
            f"({direction})"
        )


    feature_text = "\n".join(
        feature_lines
    )


    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

    system_message = """
You write ONLY the final explanation shown to the user.

Do not think aloud.
Do not describe your instructions.
Do not repeat the prompt.
Do not discuss how you generated the answer.
Do not write "Paragraph 1".
Do not write "Paragraph 2".
Do not use bullet points.
Do not use headings.
Do not write meta-commentary.

Your response must contain exactly two short paragraphs.

The FIRST characters of your response must be:

The model predicted

Paragraph 1 must explain the prediction and probability values.

Paragraph 2 must explain the influential features and why confidence is not absolute certainty.

Return ONLY those two paragraphs.
"""


    # =====================================================
    # USER PROMPT
    # =====================================================

    user_message = f"""
Generate the final user-facing explanation using ONLY
the data below.

Prediction: {prediction}

Confidence: {confidence}%

Malignant probability: {malignant_probability}%

Benign probability: {benign_probability}%

Top feature contributions:
{feature_text}

Explain the result simply for a student.

The explanation is for a machine-learning research
project. It is not a medical diagnosis.

Remember:

Start immediately with:
The model predicted

Write exactly two short paragraphs.

No headings.
No bullets.
No numbering.
No analysis.
No instructions.
No discussion of the prompt.
No meta-commentary.
"""


    # =====================================================
    # CALL AI
    # =====================================================

    try:

        response = client.chat.completions.create(

            model="openrouter/free",

            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],

            temperature=0.1,

            max_tokens=300
        )


        message = response.choices[0].message

        explanation = message.content


        if explanation:

            return clean_explanation(
                explanation
            )


    except Exception as e:

        print(
            "LLM explanation error:",
            str(e)
        )


    # =====================================================
    # FALLBACK
    # =====================================================

    return (
        f"The model predicted {prediction} with a "
        f"confidence of {confidence}%. The malignant "
        f"probability is {malignant_probability}%, while "
        f"the benign probability is {benign_probability}%. "
        f"These probabilities represent the model's "
        f"estimated output based on patterns learned from "
        f"the training data."
        "\n\n"
        f"The most influential features were identified "
        f"from the model's sensitivity to the input values. "
        f"Their contribution indicates whether changing "
        f"a feature moved the model toward the malignant "
        f"or benign class. The confidence is not absolute "
        f"certainty because machine-learning predictions "
        f"are based on learned patterns and can be incorrect."
    )


# =========================================================
# GENERAL AI CHAT
# =========================================================

def chat_with_llm(messages):

    response = client.chat.completions.create(

        model="openrouter/free",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are HEXAMIND AI, a helpful AI "
                    "assistant inside a Hybrid Quantum "
                    "Machine Learning research platform. "
                    "Answer clearly and naturally. You can "
                    "discuss programming, quantum computing, "
                    "machine learning, science, education, "
                    "and the user's project. Do not pretend "
                    "to be a medical professional or provide "
                    "medical diagnosis."
                )
            },
            *messages
        ],

        temperature=0.4,

        max_tokens=700
    )


    message = response.choices[0].message


    if message.content:

        return message.content.strip()


    return (
        "Sorry, the AI model did not return "
        "a text response."
    )