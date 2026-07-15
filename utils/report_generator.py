"""
utils/report_generator.py - Generate CSV and PDF reports from prediction history
"""
import os
import sys
import io
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import REPORTS_DIR


def generate_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return prediction history DataFrame as CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def generate_pdf_bytes(df: pd.DataFrame, stats: dict) -> bytes:
    """
    Generate a styled PDF report and return as bytes.
    Uses fpdf2 library.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return b""  # fallback

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(233, 30, 140)  # Instagram pink
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 30, "  Instagram AI Detector - Prediction Report", align="L")
    pdf.ln(10)

    # ── Generated at ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # ── Summary Stats ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_draw_color(233, 30, 140)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    stat_items = [
        ("Total Predictions", stats.get("total", 0)),
        ("Fake Accounts Detected", stats.get("fake_accounts", 0)),
        ("Bot Accounts Detected", stats.get("bots", 0)),
        ("Fake Images Detected", stats.get("fake_images", 0)),
        ("Real Accounts", stats.get("real_accounts", 0)),
        ("Human Accounts", stats.get("humans", 0)),
    ]
    for label, value in stat_items:
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(120, 9, f"  {label}", border=0, fill=True)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(70, 9, str(value), border=0, fill=True, ln=True)
        pdf.set_font("Helvetica", "", 11)
    pdf.ln(8)

    # ── Table ──────────────────────────────────────────────────────────────
    if not df.empty:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Prediction History", ln=True)
        pdf.set_draw_color(233, 30, 140)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        headers = ["Timestamp", "Module", "Prediction", "Confidence %"]
        col_w = [48, 45, 45, 42]
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(233, 30, 140)
        pdf.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_w):
            pdf.cell(w, 9, h, border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(30, 30, 30)
        for i, row in df.iterrows():
            fill = i % 2 == 0
            pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
            pdf.cell(col_w[0], 8, str(row.get("timestamp", ""))[:19], border=1, fill=fill)
            pdf.cell(col_w[1], 8, str(row.get("module", "")), border=1, fill=fill)
            pdf.cell(col_w[2], 8, str(row.get("prediction", "")), border=1, fill=fill)
            pdf.cell(col_w[3], 8, f"{row.get('confidence', 0):.1f}%", border=1, fill=fill)
            pdf.ln()
            if pdf.get_y() > 270:
                pdf.add_page()

    # ── Footer ──────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 10, "Instagram AI Detector | AI & Data Science Final Year Project", align="C")

    return bytes(pdf.output())


def save_reports(df: pd.DataFrame, stats: dict):
    """Save both CSV and PDF to the reports/ directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "report.csv")
    pdf_path = os.path.join(REPORTS_DIR, "report.pdf")

    df.to_csv(csv_path, index=False)
    pdf_bytes = generate_pdf_bytes(df, stats)
    if pdf_bytes:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    return csv_path, pdf_path
