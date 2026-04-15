# Context Window Optimizer — Setup Guide

## Files
```
chatbot_app/
├── app.py              ← Flask backend
├── requirements.txt    ← Python packages
└── static/
    └── index.html      ← Frontend UI
```

## Setup (one time)

1. Open terminal and go into the project folder:
   ```
   cd chatbot_app
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Add a free HuggingFace token for real AI responses:
   - Sign up at https://huggingface.co
   - Get free token from https://huggingface.co/settings/tokens
   - Run: `set HF_TOKEN=your_token_here`  (Windows)
   - Or:  `export HF_TOKEN=your_token_here` (Mac/Linux)
   - Without token: app uses smart mock responses (still great for demo)

## Run the app

```
python app.py
```

Then open your browser at: http://localhost:5000

## Demo script for class presentation

1. Start with Sliding Window selected
2. Send: "Tell me about hotels in Paris"
3. Send: "What flights are available from Delhi?"
4. Send: "What beaches are nice in France?"
5. Now ask: "What was the hotel name you mentioned?"
   → Sliding Window gives wrong/incomplete answer (hotel is not in last 4 msgs)

6. Switch to Full Context → sends everything, correct but high token count
7. Switch to Relevance Pruning → lowest tokens, correct answer, only hotel msg selected

Point out the Context Selection panel on the right showing exactly which messages were picked!
