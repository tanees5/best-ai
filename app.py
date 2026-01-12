from flask import Flask, request, jsonify, render_template
import os
from openai import OpenAI # Re-introduce openai import
# import google.generativeai as genai # Removed genai import
import anthropic
import requests # Import the requests library

app = Flask(__name__, static_folder='static')

# Previous OpenAI key
NVIDIA_API_KEY="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" # New NVIDIA API key
# Removed Gemini API Key
ANTHROPIC_API_KEY="sk-ant-api03-cVlTqJ_qCqA3jCqC6g6h3CqC7h7j3HqC8h8j3HqC9h9j3HqC0h0j3HqC1h1j3HqC2h2j3HqC3h3j3HqC4h4j3HqC5h5j3HqC6h6j3HqC7h7j3HqC8h8j3HqC9h9j3HqC0h0j3"
# PERPLEXITY_API_KEY="pplx-42172361212345678901234567890123456789012345678901234567890123456789012345678901234567890123456789" # Removed Perplexity API Key
MOONSHOT_API_KEY="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" # New Moonshot AI API key
# DEEPSEEK_API_KEY="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" # Removed Deepseek AI API key
GPTOSS20B_API_KEY="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" # GPT-OSS 20B API key
GPTOSS120B_API_KEY="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" # GPT-OSS 120B API key


# Configure NVIDIA (Nemotron) API client
nvidia_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

# Configure Moonshot AI client
moonshot_client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=MOONSHOT_API_KEY
)

# Removed Deepseek AI client

# Configure GPT-OSS 20B client
gptoss20b_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = GPTOSS20B_API_KEY
)

# Configure GPT-OSS 120B client
gptoss120b_client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = GPTOSS120B_API_KEY
)

# Configure Anthropic Claude API key
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Configure Perplexity AI API key
# PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions" # Removed Perplexity API URL

def simplify_summary(summary_text):
    if not GPTOSS120B_API_KEY:
        return "GPT-OSS 120B API key not configured for simplification."

    simplification_prompt = f"Please rephrase the following text to make it more user-friendly, engaging, and easy for a general audience to understand, without losing the core meaning:\n\n{summary_text}"

    try:
        simplification_completion = gptoss120b_client.chat.completions.create(
            model="openai/gpt-oss-120b", # Using GPT-OSS 120B for simplification
            messages=[
                {"role": "system", "content": "You are a helpful assistant that makes text user-friendly."},
                {"role": "user", "content": simplification_prompt}
            ],
            temperature=1,
            top_p=1,
            max_tokens=4096,
            stream=False
        )
        return simplification_completion.choices[0].message.content
    except Exception as e:
        return f"Error during simplification with GPT-OSS 120B: {e}"

def summarize_responses(question, responses):
    if not NVIDIA_API_KEY:
        return "AI service not configured for summarization."

    summary_prompt = f"""Given the user's question: '{question}' and the following information, please identify ONLY the common points, shared insights, and recurring themes. Present these common points as a single, concise paragraph.
    If no common points are found across all provided responses,and Please organize the following text into a clear and structured format with:
    - Bullet points for main ideas
    - Clear headings where appropriate
    - Numbered lists for sequential information
    - Double line breaks after each major point for better readability ,please explicitly state 'No common points found.' and provide a brief, general overview based on Nemotron's response.

"""
    for idx, (model_name, response_text) in enumerate(responses.items()):
        summary_prompt += f"**Response {idx + 1}:** {response_text}\n"
   
    try:
        summary_completion = nvidia_client.chat.completions.create(
            model="nvidia/nvidia-nemotron-nano-9b-v2", # Using Nemotron for summarization
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes and combines information from multiple sources."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
            stream=False, # Changed to False for direct summarization output
            extra_body={
                "min_thinking_tokens": 1024,
                "max_thinking_tokens": 2048
            }
        )
        return summary_completion.choices[0].message.content
    except Exception as e:
        return f"Error during summarization: {e}"


import json
import uuid

# ... (Imports remain, adding json and uuid)

# Initialize persistence
CHATS_FILE = 'chats.json'

def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_chats(chats):
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chats', methods=['GET'])
def get_chats():
    chats = load_chats()
    # Return list of {id, title} sorted by newest? (For now just dict values)
    # Sorting by some timestamp would be better, but let's just return list
    chat_list = [{"id": k, "title": v["title"]} for k, v in chats.items()]
    # Reverse to show newest created last (or first if we prepend in UI)
    return jsonify(chat_list)

@app.route('/chat/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        return jsonify(chats[chat_id])
    return jsonify({"error": "Chat not found"}), 404

@app.route('/ask', methods=['POST'])
def ask_models():
    data = request.json
    user_question = data.get('question')
    chat_id = data.get('chat_id')
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    chats = load_chats()
    
    # Create new chat if needed
    is_new_chat = False
    if not chat_id or chat_id not in chats:
        is_new_chat = True
        chat_id = str(uuid.uuid4())
        title = user_question[:30] + "..." if len(user_question) > 30 else user_question
        chats[chat_id] = {
            "title": title,
            "messages": []
        }
    
    # 1. Add User Message
    chats[chat_id]["messages"].append({"sender": "user", "text": user_question})

    final_response = ""
    used_model = "Unknown"

    # --- MASTER AI ROUTING ---
    if MOONSHOT_API_KEY:
        try:
            # 1. Classify Intent using Moonshot
            classification_prompt = f"""
            You are the Master AI Router. Analyze the following user request and classify it into exactly one of these categories:
            - ART (if the request involves generating images, describing art, or visual creativity)
            - WRITING (if the request involves creative writing, stories, essays, or editing text)
            - CODING (if the request involves programming, debugging, or technical explanation of code)
            - OTHER (for anything else)
            
            User Request: "{user_question}"
            
            Reply ONLY with the category name (ART, WRITING, CODING, or OTHER).
            """
            
            router_completion = moonshot_client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1, # Low temp for determinism
                max_tokens=10
            )
            intent = router_completion.choices[0].message.content.strip().upper()
            print(f"DEBUG: Intent classified as {intent}") # Useful for server logs

            # 2. Route based on Intent
            if "ART" in intent:
                # Use NVIDIA (Nemotron) for Art
                used_model = "NVIDIA (Art)"
                if NVIDIA_API_KEY:
                    nvidia_completion = nvidia_client.chat.completions.create(
                        model="nvidia/nvidia-nemotron-nano-9b-v2",
                        messages=[{"role": "user", "content": user_question}],
                        temperature=0.7,
                        max_tokens=2048,
                        extra_body={"min_thinking_tokens": 0,"max_thinking_tokens": 0} # Disable thinking for speed or enable per preference
                    )
                    final_response = nvidia_completion.choices[0].message.content
                else:
                    final_response = "NVIDIA API Key missing for Art task."

            elif "WRITING" in intent:
                # Use GPT-OSS 120B for Writing
                used_model = "GPT-OSS 120B (Writing)"
                if GPTOSS120B_API_KEY:
                    gpt_completion = gptoss120b_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[{"role": "user", "content": user_question}],
                        temperature=0.9, # Higher creative
                        max_tokens=4096
                    )
                    final_response = gpt_completion.choices[0].message.content
                else:
                    final_response = "GPT-OSS 120B API Key missing for Writing task."

            elif "CODING" in intent:
                # Use Moonshot itself for Coding
                used_model = "Moonshot (Coding)"
                # Reuse client
                params = {
                    "model": "moonshotai/kimi-k2-instruct-0905",
                    "messages": [{"role": "user", "content": user_question}],
                    "temperature": 0.2, # Low temp for code
                    "max_tokens": 4096
                }
                moonshot_exec = moonshot_client.chat.completions.create(**params)
                final_response = moonshot_exec.choices[0].message.content

            else:
                # Default to Moonshot for General/Other
                used_model = "Moonshot (General)"
                params = {
                    "model": "moonshotai/kimi-k2-instruct-0905",
                    "messages": [{"role": "user", "content": user_question}],
                    "temperature": 0.5,
                    "max_tokens": 4096
                }
                moonshot_exec = moonshot_client.chat.completions.create(**params)
                final_response = moonshot_exec.choices[0].message.content

        except Exception as e:
            final_response = f"Master AI Error: {e}"
            print(f"Error: {e}")
    else:
        final_response = "Master AI (Moonshot) Key not configured."
    
    # 2. Add AI Message
    chats[chat_id]["messages"].append({"sender": "ai", "text": final_response})
    
    # Save
    save_chats(chats)

    return jsonify({
        "combined_response": final_response, # Frontend expects this key
        "chat_id": chat_id,
        "title": chats[chat_id]["title"] if is_new_chat else None,
        "used_model": used_model # Optional info if we want to show it later
    })

if __name__ == '__main__':
    app.run(debug=True)

