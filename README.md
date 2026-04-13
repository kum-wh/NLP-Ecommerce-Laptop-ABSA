# Sentiment Analysis Chatbot For Ecommerce

## System Deployment

The system is deployed as an Gradio application hosted on Hugging Face Spaces, 
The live demo is available at : https://huggingface.co/spaces/whismyswift/NLP_Amazon_Laptop_Reviews_ChatBot 

### To deploy locally:

1) Clone this repository
2) Install dependencies via pip install requirement.txt
3) Run python app.py to start the application

## Models

All Finetuned models are uploaded to Hugging Face Hub and can be downloaded directly.

- Deberta ABSA Model: https://huggingface.co/whismyswift/deberta-absa 
- T5 ABSA Model: https://huggingface.co/whismyswift/t5-absa-2 
- Review Summary Model: https://huggingface.co/whismyswift/BART_Summary 

## Json Files
- meta_electroics.json: Contains all the products in the electronics category and their respectives information. Used to select products with high number of reviews to focus on.
- electrionics.json: Contains all the reviews of all electronic products
- clean_electronics.json: Contains the reviews of the selected products
- dataset_text_labels.json: Sememval Test Dataset with Generated Sentiment Labels for ABSA fine Tuning
- selected_reviews.json: Randomly selected relevant reviews for predefined queries.
- generated_summaries.json: Generated Summary Dataset based on the sets of reviews in selected_reviews.json

## Python Files
- NLP_dataset: Use for manipulation of the dataset
- NLP_ABSA: Perform the ABSA FineTuning of DeBERTa and T5 Model
- NLP_Aspect_Term_Extraction: Perform the ATE of DeBERTa and T5 Model (Not Used in Final System)
- NLP_Summary: Perform the Summary FineTuning of BART Model
- NLP_System:  Integrating the different components and models into the system
- NLP_Frontend: The Gradio Code for Frontend, same as app.py