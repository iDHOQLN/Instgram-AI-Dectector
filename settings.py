"""
settings.py - Streamlit page & theme configuration helpers
"""
import streamlit as st
import os

# ── Resolve CSS paths relative to this file ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../config/
_ROOT = os.path.dirname(_HERE)                              # project root

# Primary: config/style.css  |  Fallback: assets/style.css
_CSS_PRIMARY  = os.path.join(_HERE,  "style.css")
_CSS_FALLBACK = os.path.join(_ROOT, "assets", "style.css")


def configure_page(title: str = "Instagram AI Detector", icon: str = "🔍", layout: str = "wide"):
    """Apply consistent page configuration."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )


def inject_custom_css():
    """Load and inject the custom CSS stylesheet.

    Tries config/style.css first; falls back to assets/style.css.
    """
    css_path = _CSS_PRIMARY if os.path.exists(_CSS_PRIMARY) else _CSS_FALLBACK
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def sidebar_branding():
    """Render the branded sidebar header."""
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <h2 style='color:#E91E8C; font-size:1.4rem; margin:0;'>🔍 Instagram AI</h2>
        <p style='color:#aaa; font-size:0.75rem; margin:4px 0 0 0;'>Fake Detection System v1.0.1</p>
    </div>
    """, unsafe_allow_html=True)
