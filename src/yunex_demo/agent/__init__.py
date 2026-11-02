"""An agent: the model picks the tools, and code runs and guards every one of them.

No Streamlit in this package. ``hooks`` holds the guards, ``tools`` what they protect,
``loop`` the conversation with the model, and ``approval`` the human's decision.
"""
