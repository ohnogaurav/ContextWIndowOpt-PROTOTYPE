from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import time
import requests
import os
import json

app = Flask(__name__, static_folder='static')

HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
EMBED_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

conversation_history = []

def get_embeddings(texts):
    try:
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        resp = requests.post(EMBED_API_URL, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    np.random.seed(42)
    return [np.random.rand(384).tolist() for _ in texts]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def count_tokens(text):
    return max(1, len(text.split()) + len(text) // 6)

def get_llm_response(context_messages, query):
    try:
        prompt = ""
        for m in context_messages:
            role = "User" if m["role"] == "user" else "Bot"
            prompt += f"{role}: {m['content']}\n"
        prompt += f"User: {query}\nBot:"
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 80, "temperature": 0.7, "do_sample": True},
            "options": {"wait_for_model": True}
        }
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and result:
                generated = result[0].get("generated_text", "")
                if "Bot:" in generated:
                    answer = generated.split("Bot:")[-1].strip()
                    answer = answer.split("User:")[0].strip()
                    if answer:
                        return answer
    except Exception:
        pass
    return generate_mock_response(query, context_messages)

def generate_mock_response(query, context_messages):
    q = query.lower()
    history_text = " ".join(m["content"].lower() for m in context_messages)
    
    if any(w in q for w in ["hotel", "stay", "accommodation"]):
        if "lumiere" in history_text or "hotel" in history_text:
            return "Based on our conversation, the hotel I mentioned was Hotel Lumière on Rue de Rivoli in Paris — very central and quiet."
        return "I'd recommend Hotel Lumière on Rue de Rivoli. It's central, quiet, and well-reviewed."
    if any(w in q for w in ["flight", "fly", "airline"]):
        return "For flights, Air France and IndiGo both have good direct routes. Flight time is roughly 8–9 hours."
    if any(w in q for w in ["food", "eat", "restaurant", "cuisine"]):
        return "French cuisine is world-class! Try croissants and café au lait for breakfast, and crêpes for a quick snack."
    if any(w in q for w in ["beach", "coast", "sea"]):
        return "The French Riviera is stunning — Nice and Cannes have beautiful beaches. Best visited May–September."
    if any(w in q for w in ["weather", "temperature", "climate"]):
        return "Paris is best in spring (April–June) and fall. Summers are warm around 25°C, winters can be cold and grey."
    if any(w in q for w in ["first", "earlier", "mentioned", "said", "remember", "what was"]):
        if context_messages:
            return f"Looking back at our conversation: {context_messages[0]['content'][:80]}..."
        return "I don't have enough context to recall that. Could you give me more detail?"
    return "That's a great question! Based on our conversation, I'd be happy to help you plan your trip. What specific aspect would you like to know more about?"

def select_full_context(history):
    return history, list(range(len(history)))

def select_sliding_window(history, k=4):
    selected = history[-k:] if len(history) > k else history
    indices = list(range(max(0, len(history) - k), len(history)))
    return selected, indices

def select_relevance_pruning(history, query, top_k=3):
    if not history:
        return [], []
    texts = [m["content"] for m in history]
    all_texts = texts + [query]
    embeddings = get_embeddings(all_texts)
    query_emb = embeddings[-1]
    hist_embs = embeddings[:-1]
    scores = [cosine_similarity(query_emb, e) for e in hist_embs]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_indices_sorted = sorted(top_indices)
    selected = [history[i] for i in top_indices_sorted]
    return selected, top_indices_sorted

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'full')
    
    start_time = time.time()
    
    if method == 'full':
        context, selected_indices = select_full_context(conversation_history)
    elif method == 'sliding':
        context, selected_indices = select_sliding_window(conversation_history, k=4)
    else:
        context, selected_indices = select_relevance_pruning(conversation_history, query, top_k=3)
    
    response = get_llm_response(context, query)
    elapsed = round(time.time() - start_time, 2)
    
    context_text = " ".join(m["content"] for m in context) + " " + query
    tokens_used = count_tokens(context_text)
    
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response})
    
    return jsonify({
        "response": response,
        "tokens_used": tokens_used,
        "time_taken": elapsed,
        "selected_indices": selected_indices,
        "total_messages": len(conversation_history) - 2,
        "method": method
    })

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"history": conversation_history})

@app.route('/clear', methods=['POST'])
def clear_history():
    conversation_history.clear()
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
