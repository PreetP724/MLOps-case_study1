import gradio as gr
from huggingface_hub import InferenceClient
import os

pipe = None
stop_inference = False

# Fancy styling
fancy_css = """
#main-container {
    background-color: #3498db;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
    use_local_model: bool,
):
    
    global pipe

    # Build messages from history

    few_shot_examples = [
    {'role': 'user', 'content': 'I do not understand my homework.'},
    {'role': 'assistant', 'content': "That's okay! This is all part of the learning process. Let's take a look and figure it out together!"},
     {'role': 'user', 'content': 'What does the word responsible mean?'},
    {'role': 'assistant', 'content': "That's an amazing question and certainly a tough word! Responsible means doing things that you have to or should do. Let's see what it looks like in a sentence: you are responsible for doing your homework. This makes sense because your homework is something you have to do." },
     {'role': 'user', 'content': "What is 2 + 3?"},
    {'role': 'assistant', 'content': "Let's use our fingers to count this out! First, start with 2 by putting up two fingers. Now, put up three more fingers to add 3. Count how many figures you are holding up. The answer is 5!"}
]
    messages = [{"role": "system", "content": system_message}]
    messages.extend(few_shot_examples)
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = ""

    if use_local_model:
        print("[MODE] local")
        from transformers import pipeline
        import torch 
        if pipe is None:
            pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

        # Build prompt as plain text
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt):]
        yield response.strip()

    else:
        print("[MODE] api")

        if hf_token is None or not getattr(hf_token, "token", None):
            yield "⚠️ Please log in with your Hugging Face account first."
            return

        client = InferenceClient(token=hf_token.token, model="openai/gpt-oss-20b")



        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p= top_p,
        ):
            choices = chunk.choices
            token = ""
            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content
            response += token
            yield response


chatbot = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value= "You are a helpful learning assistant for a children in elementary school in the age range of 6-10. You talk nicely and use simple words like you would to a kid an elementary school. Answer questions so a kid in elementary school would understand and enjoy. Take complex topics and turn them into simple ones. Be relatable to a kid in elementary school. Be encouraging.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
        gr.Checkbox(label="Use Local Model", value=False),
    ],
    
)

with gr.Blocks(css=fancy_css) as demo:
    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>🌟 Learning Assistant 🌟</h1>")
        gr.LoginButton()
    chatbot.render()

if __name__ == "__main__":
    demo.launch()
