import gradio as gr
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

max_input_length = 584
max_target_length = 128

# ──────────────────────────────────────────────
# 1. Load reviews from local JSON
# ──────────────────────────────────────────────
PRODUCT_NAMES = {
    "B0C2ZMJW53": "Acer Aspire 3 Slim Laptop",
    "B09P29VXG1": "SAMSUNG Galaxy Tab A8 10.5 Android Tablet",
    "B096JQY4YR": "HP Chromebook 14-inch HD Touchscreen Laptop",
    "B0869L1326": "ASUS VivoBook 15 Thin and Light Laptop",
    "B0BBKNHDRH": "Acer Nitro 5 Gaming Laptop",
    "B08BH87FPJ": "Lenovo Tab M10 Plus Tablet",
    "B0BLBY439T": "Acer Aspire 5 A515-46-R14K Slim Laptop",
    "B0C7H1XD7V": "Lenovo Flex 5 14 2-in-1 Laptop",
    "B07GM2J11Q": "Lenovo Chromebook C330 2-in-1 Convertible Laptop",
    "B0B72PYMGX": "Lenovo IdeaPad 3 11 Chromebook Laptop"
}

with open("clean_electronics.json", "r") as f:
    DATA = json.load(f)

PRODUCTS = {}

for item in DATA:
    product_id = item.get("parent_asin")
    if product_id in PRODUCTS:
        PRODUCTS[product_id].append(item['text'])
    else:
        PRODUCTS[product_id] = [item['text']]

# ──────────────────────────────────────────────
# 2. Build RAG vectorstores
# ──────────────────────────────────────────────

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

all_documents = []  
for product_id, reviews in PRODUCTS.items():
    documents = [
        Document(
            page_content=r,
            metadata={"product": product_id}
        )
        for r in reviews
    ]
    all_documents.extend(documents)

chunks = text_splitter.split_documents(all_documents)
vectorstore = Chroma.from_documents(chunks, embeddings)

print(f"Vectorstore ready")

# ──────────────────────────────────────────────
# 3. Load BART model
# ──────────────────────────────────────────────

print("Loading BART model ...")
tokenizer = AutoTokenizer.from_pretrained("whismyswift/BART_Summary")
model = AutoModelForSeq2SeqLM.from_pretrained("whismyswift/BART_Summary", dtype="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"BART model loaded.")

# ──────────────────────────────────────────────
# 4. Build context string from reviews
# ──────────────────────────────────────────────

def preprocess_single_example(example):
    instruction = "Summarize customer reviews about the specific aspect mentioned in the query."
    r_processed = " ".join(example['reviews_input']) if isinstance(example['reviews_input'], list) else str(example['reviews_input'])
    input_text = f"instruction: {instruction} Query: {example['question']} Reviews: {r_processed}"
    model_inputs = tokenizer(input_text, max_length=max_input_length, truncation=True, return_tensors="pt")
    return model_inputs


# ──────────────────────────────────────────────
# 4. Chat function — runs BART locally
# ──────────────────────────────────────────────

def chat(message, history, product_name):
    if not product_name:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please select a product first."}
        ]
    
    similar_reviews = vectorstore.similarity_search(message, k=5, filter={"product": product_name})
    
    sample_input = {
        "question": message,
        "reviews_input": [similar_reviews[0].page_content,
                        similar_reviews[1].page_content,
                        similar_reviews[2].page_content,
                        similar_reviews[3].page_content,
                        similar_reviews[4].page_content]
    }

    processed_sample = preprocess_single_example(sample_input)

    # Move to GPU if available
    input_ids = processed_sample['input_ids'].to(model.device)
    attention_mask = processed_sample['attention_mask'].to(model.device)

    # Generate summary
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True
        )

    decoded_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": decoded_summary},
    ]


# ──────────────────────────────────────────────
# 5. Reset chat when product changes
# ──────────────────────────────────────────────

def reset_chat():
    welcome = f"Hi! Ask me anything about the product based on its customer reviews."
    return [{"role": "assistant", "content": welcome}]


# ──────────────────────────────────────────────
# 5. Gradio UI
# ──────────────────────────────────────────────

with gr.Blocks(
    title="Product Review Chat",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
        #header { text-align:center; padding:24px 0 8px 0; }
        #header h1 { font-size:2rem; margin-bottom:4px; }
        #header p  { color:#666; margin:0; }
    """
) as demo:

    gr.HTML("""
        <div id="header">
          <h1>Product Review Assistant</h1>
          <p>Select a product and chat with an AI trained on its customer reviews</p>
        </div>
    """)

    dropdown = gr.Dropdown(
        choices=list(PRODUCT_NAMES.values()),
        label="Choose a Product",
        value=list(PRODUCT_NAMES.keys())[0] if PRODUCT_NAMES else None,
        interactive=True,
    )

    chatbot = gr.Chatbot(
        label="Chat",
        type="messages",
        height=450,
        show_copy_button=True,
    )

    msg_input = gr.Textbox(
        placeholder="Ask something about this product...",
        label="Your message",
        lines=1,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    # On load — show welcome message for first product
    demo.load(fn=reset_chat, inputs=[], outputs=[chatbot])

    # Product change — reset chat with new welcome message
    dropdown.change(fn=reset_chat, inputs=[], outputs=[chatbot])

    # Send message — stream response
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, dropdown],
        outputs=[chatbot],
    ).then(lambda: "", outputs=[msg_input])

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, dropdown],
        outputs=[chatbot],
    ).then(lambda: "", outputs=[msg_input])

    # Clear chat
    clear_btn.click(fn=reset_chat, inputs=[], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()
