import gradio as gr

def greet(name):
    return f"Hello {name}! Your Geographic RAG System is working! 🌍"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Enter your name"),
    outputs=gr.Textbox(label="Response"),
    title="🌍 Geographic RAG System - Test",
    description="Testing if the Space is working correctly"
)

if __name__ == "__main__":
    demo.launch() 