PROMPT_FORMAT = """
THIS IS THE CONTEXT:

{}

__________

Based on this context, try to answer the following question:
{}

__________

If the answer cannot be found in the context, just say "I don't know."

"""


SYS_PROMPT = """
You are RAG GPT
Your goal is to provide answers to questions based on the context provided.

You can say "I don't know" if you cannot find the answer in the context.


"""
