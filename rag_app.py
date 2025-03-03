#pip install PyPDF2 transformers sentence-transformers faiss-cpu

import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

# Step 1: Extract text from a PDF file
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Step 2: Create a FAISS index for retrieval
def create_faiss_index(text_chunks, embedder):
    embeddings = embedder.encode(text_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

# Step 3: Retrieve relevant chunks
def retrieve_relevant_chunks(query, index, embedder, text_chunks, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [text_chunks[idx] for idx in indices[0]]

# Step 4: Generate an answer using a language model
def generate_answer(query, context, tokenizer, model):
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# GUI Application
class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Application")
        self.root.geometry("600x400")

        # Load models
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Smaller generative model
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")  # Smaller generative model

        # Variables
        self.file_path = None
        self.text_chunks = None
        self.index = None

        # GUI Components
        self.label = tk.Label(root, text="RAG Application", font=("Arial", 16))
        self.label.pack(pady=10)

        self.load_button = tk.Button(root, text="Load File", command=self.load_file)
        self.load_button.pack(pady=10)

        self.query_label = tk.Label(root, text="Enter your question:")
        self.query_label.pack(pady=5)

        self.query_entry = tk.Entry(root, width=50)
        self.query_entry.pack(pady=5)

        self.ask_button = tk.Button(root, text="Ask", command=self.ask_question)
        self.ask_button.pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(root, width=70, height=10)
        self.output_text.pack(pady=10)

    # Load file and process it
    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf"), ("Text Files", "*.txt")])
        if not self.file_path:
            messagebox.showerror("Error", "No file selected!")
            return

        if self.file_path.endswith(".pdf"):
            text = extract_text_from_pdf(self.file_path)
        elif self.file_path.endswith(".txt"):
            with open(self.file_path, "r") as file:
                text = file.read()
        else:
            messagebox.showerror("Error", "Unsupported file format! Please provide a PDF or TXT file.")
            return

        # Split text into chunks
        self.text_chunks = text.split(". ")  # Split by sentences for simplicity
        self.index = create_faiss_index(self.text_chunks, self.embedder)
        messagebox.showinfo("Success", f"File loaded successfully! Processed {len(self.text_chunks)} text chunks.")

    # Ask a question and display the answer
    def ask_question(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please load a file first!")
            return

        query = self.query_entry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a question!")
            return

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query, self.index, self.embedder, self.text_chunks)
        context = " ".join(relevant_chunks)

        # Generate and display the answer
        answer = generate_answer(query, context, self.tokenizer, self.model)
        self.output_text.insert(tk.END, f"Question: {query}\nAnswer: {answer}\n\n")
        self.query_entry.delete(0, tk.END)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
