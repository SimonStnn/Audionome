import streamlit as st

st.set_page_config(
    page_title="Audionome - Documentation",
    page_icon="📄",
)

# Read ../README.md
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

st.write(readme)
