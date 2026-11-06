import json
import os
import re
import time
from groq import Groq
import pandas as pd
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from a PDF file."""
    reader = PdfReader(pdf_path, strict=False)
    text_list = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_list)


def split_text_into_chunks(text: str, chunk_size: int = 15000) -> list:
    """Splits text string into smaller portions to avoid input rate limits."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def parse_json_safely(response_text: str) -> list:
    """Cleans up markdown formatting wraps and extracts JSON content securely."""
    try:
        # Remove markdown code block wrappers if the model includes them
        clean_text = re.sub(r"```json\s*|\s*```", "", response_text.strip())
        data = json.loads(clean_text)
        # Ensure we return a list of dictionaries
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Skipping a section: Failed to parse JSON response. Error: {e}")
        return []


def analyze_pdf_and_save_to_csv(
    pdf_path: str, user_prompt: str, csv_path: str = "rule.csv"
):
    """Extracts rules into structured JSON format via Groq and appends them to a master CSV."""
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(pdf_text)
    print(f"Document split into {len(chunks)} chunks to fit the rate limit...")

    client = Groq(api_key="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")
    all_rules = []

    # A tighter, low-token system instruction ensuring it knows it must output a JSON list
    json_instruction_schema = (
        "Return a valid JSON array of objects. Do not write text outside the JSON"
        ' array. Use this schema: [{"Section": "string", "Rule_Title": "string",'
        ' "Details": "short description"}]'
    )

    for index, chunk in enumerate(chunks):
        print(f"Processing chunk {index + 1}/{len(chunks)}...")

        prompt = f"""
            Extract requirements targeting: {user_prompt}
            
            [START OF DOCUMENT PART {index + 1}]
            {chunk}
            [END OF DOCUMENT PART {index + 1}]

            Schema Constraint: {json_instruction_schema}
            """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a technical data parser that outputs a valid JSON"
                        " list array of objects. Keep details extremely brief to"
                        " prevent cutoffs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model="qwen/qwen3.8-27b",
            max_tokens=500,
            # ⬇️ FORCES THE GROQ ENGINE TO KEEP JSON STRUCTURE VALID ⬇️
            response_format={"type": "json_object"},
        )

        # Extract response text
        response_content = response.choices[0].message.content

        # If the model wraps the array in a parent dictionary object, handle it smoothly
        # Native JSON mode sometimes wraps arrays like {"rules": [...]}.
        try:
            parsed_json = json.loads(response_content)
            if isinstance(parsed_json, dict):
            # Extract the first list found inside the dictionary values
                for value in parsed_json.values():
                    if isinstance(value, list):
                        all_rules.extend(value)
                        break
                else:
                    all_rules.append(parsed_json)
            elif isinstance(parsed_json, list):
                all_rules.extend(parsed_json)
        except Exception as parse_error:
            print(f"Skipping chunk {index + 1} due to parsing issue: {parse_error}")

    except Exception as api_error:
        print(f"API Error on chunk {index + 1}: {api_error}")

    if index < len(chunks) - 1:
        time.sleep(2)

    # 4. Turn structured records into a Pandas DataFrame and save
    if all_rules:
        new_df = pd.DataFrame(all_rules)

        # Clean up column names just in case the model used slightly different casing
        expected_cols = ["Section", "Rule_Title", "Details"]
        for col in expected_cols:
            if col == "Section" and "section" in new_df.columns:
                new_df.rename(columns={"section": "Section"}, inplace=True)
            if col == "Rule_Title" and "rule_title" in new_df.columns:
                new_df.rename(columns={"rule_title": "Rule_Title"}, inplace=True)
            if col == "Details" and "details" in new_df.columns:
                new_df.rename(columns={"details": "Details"}, inplace=True)

        # Keep only the valid columns we need
        new_df = new_df.reindex(columns=expected_cols).dropna(subset=["Rule_Title"])

        if os.path.exists(csv_path):
            try:
                # Check if the file size is 0 bytes to prevent parsing crashes
                if os.path.getsize(csv_path) == 0:
                    raise pd.errors.EmptyDataError

                existing_df = pd.read_csv(csv_path)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(csv_path, index=False)
                print(
                    f"\nSuccessfully appended {len(new_df)} new items to master data"
                    f" file: '{csv_path}'"
                )
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # If the file is empty or corrupted, overwrite it as a fresh file
                new_df.to_csv(csv_path, index=False)
                print(
                    f"\nOverwrote empty/corrupted file and saved {len(new_df)} items"
                    f" to: '{csv_path}'"
                )
        else:
            # If the file does not exist yet, create it from scratch
            new_df.to_csv(csv_path, index=False)
            print(
                f"\nCreated a brand new master file and saved {len(new_df)} items"
                f" to: '{csv_path}'"
            )
    else:
        print("\nNo valid structured rules were parsed during this run.")



if __name__ == "__main__":
    file_path = r"C:\Users\ROG\Downloads\Telegram Desktop\Robocon_2027_Rulebook_v1-1.pdf"
    target_extraction_prompt = "Summary the RuleBook."

    print("Starting structured parsing routine pipeline...")
    try:
        analyze_pdf_and_save_to_csv(file_path, target_extraction_prompt, "rule.csv")
    except Exception as e:
        print(f"\nAn structural error occurred: {e}")
