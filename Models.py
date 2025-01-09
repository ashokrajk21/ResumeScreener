import openai
import gensim
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import streamlit as st

# Set OpenAI API key
openai.api_key = "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"  # Replace with your actual API key

# Ensure 'punkt' is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' resource...")
    nltk.download('punkt')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@st.cache_resource
def get_HF_embeddings(sentences, model_name):
    """
    Generate embeddings using a HuggingFace model.
    :param sentences: List of sentences to embed.
    :param model_name: Name of the HuggingFace model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).numpy()

@st.cache_resource
def get_gpt3_embeddings(sentences):
    """
    Generate embeddings using OpenAI's GPT-3.5 embeddings model.
    :param sentences: List of sentences to embed.
    """
    embeddings = []
    for sentence in sentences:
        response = openai.Embedding.create(
            input=sentence,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

@st.cache_data
def get_doc2vec_embeddings(JD, text_resume):
    """
    Generate embeddings using Doc2Vec.
    :param JD: Job description text.
    :param text_resume: List of resumes' text.
    """
    nltk.download('punkt', quiet=True)  # Ensure punkt is downloaded silently
    
    # Prepare tagged data
    tagged_data = [TaggedDocument(words=word_tokenize(JD.lower()), tags=["JD"])]
    for idx, resume_text in enumerate(text_resume):
        tagged_data.append(TaggedDocument(words=word_tokenize(resume_text.lower()), tags=[f"RESUME_{idx}"]))
    
    # Train Doc2Vec model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=3, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=80)
    
    # Get embeddings
    JD_embeddings = model.dv["JD"].reshape(1, -1)
    resume_embeddings = [model.dv[f"RESUME_{idx}"].reshape(1, -1) for idx in range(len(text_resume))]
    return JD_embeddings, resume_embeddings

def cosine(embeddings1, embeddings2):
    """
    Calculate cosine similarity between two sets of embeddings.
    """
    scores = []
    for emb1 in embeddings1:
        similarity = cosine_similarity(np.array(emb1).reshape(1, -1), np.array(embeddings2))
        scores.append(similarity[0][0] * 100)
    return scores
