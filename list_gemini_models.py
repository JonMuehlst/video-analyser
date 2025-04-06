import os
import litellm
import json
import dotenv

# Load environment variables (like GEMINI_API_KEY) from .env file if present
dotenv.load_dotenv()

# Set LiteLLM verbosity to see more details if needed
# litellm.set_verbose = True

print("Attempting to list models from Google AI Studio (Gemini API)...")

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    print("\nERROR: GEMINI_API_KEY environment variable not found.")
    print("Please set the GEMINI_API_KEY environment variable before running this script.")
else:
    try:
        # Use litellm.get_model_list - it tries to fetch models from providers
        # We filter for 'google' which covers both Vertex and AI Studio in litellm
        # LiteLLM might not have a direct way to *only* list AI Studio models via get_model_list
        # but we can try calling the generic function and see what it returns, or
        # make a direct API call simulation if needed. Let's try get_model_list first.

        # Option 1: Try litellm.get_model_list (might include Vertex models too)
        print("\nAttempting listing via litellm.get_model_list...")
        # Note: get_model_list might require different auth or might not specifically filter by provider easily.
        # It often lists models LiteLLM *knows* about, not necessarily what *your key* can access.
        # model_list = litellm.get_model_list(api_key=api_key) # This might not work as expected for specific provider listing

        # Option 2: A more direct approach is to make a minimal completion call
        # to a known *working* model and see if the 'list models' hint comes up on error,
        # or better, use the underlying google-generativeai library if installed.

        # Option 3: Using google-generativeai directly (if available in the user's env)
        # This is the most reliable way to list models for Google AI Studio specifically.
        try:
            import google.generativeai as genai

            print("\nAttempting listing via google-generativeai library...")
            genai.configure(api_key=api_key)

            print("\nAvailable models (supporting generateContent):")
            model_count = 0
            for m in genai.list_models():
              if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
                model_count += 1
            if model_count == 0:
                print("  (No models found supporting generateContent)")

        except ImportError:
            print("\n'google-generativeai' library not found.")
            print("Cannot list models directly. Please install it (`pip install google-generativeai`)")
            print("or rely on the error messages from your main script.")
        except Exception as e:
            print(f"\nError listing models using google-generativeai: {e}")


    except litellm.exceptions.AuthenticationError as e:
        print(f"\nLiteLLM Authentication Error: {e}")
        print("Please ensure your GEMINI_API_KEY is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("This might be due to network issues, incorrect API key, or litellm issues.")

print("\nScript finished.")