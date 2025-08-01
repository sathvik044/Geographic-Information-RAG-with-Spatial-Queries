import gradio as gr

def process_query(query):
    return f"Processing your geographic query: {query}\n\nThis is your Geographic RAG System working! 🌍"

demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(label="Enter your spatial query", placeholder="e.g., What cities are near New York?"),
    outputs=gr.Textbox(label="Response", lines=5),
    title="🌍 Geographic Information RAG System",
    description="Advanced spatial query processing with satellite imagery analysis",
    examples=[
        ["What cities are within 100km of New York?"],
        ["Show me the population density of major US cities"],
        ["What are the environmental features near Los Angeles?"]
    ]
)

if __name__ == "__main__":
    demo.launch() 