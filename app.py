import streamlit as st
from typing import List, Dict
from openai import OpenAI

# ---------------------------
# Configuration and Utilities
# ---------------------------

LANGUAGES = [
    "Hindi",
    "Bengali",
    "Telugu",
    "Marathi",
    "Tamil",
    "Gujarati",
    "Kannada",
    "Malayalam",
    "Punjabi (Gurmukhi)",
    "Odia",
    "Assamese",
    "Urdu",
]

MODEL_OPTIONS = ["gpt-4", "gpt-3.5-turbo"]


def get_language_instruction(language: str) -> str:
    if "Punjabi" in language:
        return "Punjabi written in Gurmukhi script"
    if "Urdu" in language:
        return "Urdu using the Perso-Arabic script"
    return language


def build_user_prompt(language_instruction: str, english_text: str) -> str:
    return (
        f"Task: Translate the following English text into {language_instruction}.\n"
        f"Requirements:\n"
        f"- Preserve meaning, tone, and politeness.\n"
        f"- Use native, natural phrasing in {language_instruction}.\n"
        f"- Do not include any explanations, notes, language names, or quotes.\n"
        f"- Return only the translation text.\n\n"
        f"English text:\n{english_text.strip()}"
    )


def translate_text(client: OpenAI, model: str, english_text: str, language: str) -> str:
    try:
        language_instruction = get_language_instruction(language)
        user_prompt = build_user_prompt(language_instruction, english_text)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[Error translating to {language}: {e}]"


def translate_to_multiple(
    client: OpenAI, model: str, english_text: str, languages: List[str]
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for lang in languages:
        results[lang] = translate_text(client, model, english_text, lang)
    return results


# ---------------------------
# Streamlit App
# ---------------------------

def main():
    st.set_page_config(page_title="Indian Languages Translator", page_icon="🌐", layout="centered")
    st.title("🌐 Indian Languages Translator")
    st.caption("Translate English text into multiple Indian regional languages. Powered by OpenAI.")

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Model", MODEL_OPTIONS, index=0)
        st.markdown(
            "Note: Set your OpenAI API key in the environment variable `OPENAI_API_KEY` before running."
        )
        st.divider()
        st.write("Tips:")
        st.write("- Keep input concise for faster results.")
        st.write("- Choose fewer languages to reduce latency.")

    if "history" not in st.session_state:
        st.session_state.history = []

    english_text = st.text_area(
        "Enter English text",
        height=150,
        placeholder="Type or paste English text here...",
    )

    selected_languages = st.multiselect(
        "Select target languages",
        LANGUAGES,
        default=["Hindi", "Tamil", "Telugu"],
        help="Choose one or more Indian languages for translation.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        translate_btn = st.button("Translate", type="primary")
    with col2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.session_state.history = []
        st.experimental_rerun()

    if translate_btn:
        if not english_text.strip():
            st.warning("Please enter some English text to translate.")
        elif not selected_languages:
            st.warning("Please select at least one target language.")
        else:
            client = OpenAI()
            with st.spinner("Translating..."):
                results = translate_to_multiple(client, model, english_text, selected_languages)

            st.subheader("Translations")
            for lang in selected_languages:
                st.markdown(f"**{lang}**")
                st.write(results.get(lang, ""))

            st.session_state.history.insert(
                0,
                {
                    "text": english_text.strip(),
                    "model": model,
                    "results": {lang: results.get(lang, "") for lang in selected_languages},
                },
            )

    if st.session_state.history:
        st.divider()
        st.subheader("History")
        for idx, item in enumerate(st.session_state.history[:5], start=1):
            with st.expander(f"Request {idx} • Model: {item['model']}"):
                st.markdown("Original English:")
                st.write(item["text"])
                st.markdown("---")
                st.markdown("Translations:")
                for lang, tr in item["results"].items():
                    st.markdown(f"- {lang}:")
                    st.write(tr)


if __name__ == "__main__":
    main()