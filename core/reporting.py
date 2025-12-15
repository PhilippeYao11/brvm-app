import io
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_portfolio_report_pdf(
    title: str,
    date_range: str,
    asset_stats: pd.DataFrame,
    weights_df: pd.DataFrame,
    equity_curve: pd.Series,
    sim_chart_df: pd.DataFrame,   # columns: Date, Value, Type
    risk_row: pd.Series,          # one row of df_risk (Series)
    technical_rules_text: str,
) -> bytes:
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(date_range, styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # 1) Universe table
    story.append(Paragraph("1. Universe overview", styles["Heading2"]))
    tbl_df = asset_stats.copy()
    tbl_df = tbl_df.reset_index().rename(columns={"index": "Asset"})
    cols = [c for c in ["Asset", "mu_ann", "sigma_ann", "mu_3m"] if c in tbl_df.columns]
    tbl_df = tbl_df[cols].fillna(0.0)

    table_data = [tbl_df.columns.tolist()] + tbl_df.round(6).values.tolist()
    t = Table(table_data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # 2) Weights chart
    story.append(Paragraph("2. Optimised portfolio", styles["Heading2"]))

    fig = plt.figure()
    w = weights_df["Weight"].sort_values(ascending=True)
    plt.barh(w.index.astype(str), w.values)
    plt.title("Portfolio weights")
    img = Image(io.BytesIO(_fig_to_png_bytes(fig)))
    img.drawHeight = 8 * cm
    img.drawWidth = 16 * cm
    story.append(img)
    story.append(Spacer(1, 0.3 * cm))

    # Equity curve chart
    fig = plt.figure()
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title("Historical portfolio value")
    plt.xticks(rotation=30)
    img = Image(io.BytesIO(_fig_to_png_bytes(fig)))
    img.drawHeight = 7 * cm
    img.drawWidth = 16 * cm
    story.append(img)
    story.append(Spacer(1, 0.5 * cm))

    # 3) Projections chart
    story.append(Paragraph("3. Projections & risk", styles["Heading2"]))
    fig = plt.figure()
    for typ in sim_chart_df["Type"].unique():
        sub = sim_chart_df[sim_chart_df["Type"] == typ]
        plt.plot(sub["Date"], sub["Value"], label=str(typ))
    plt.title("Portfolio value: historical + projected mean")
    plt.xticks(rotation=30)
    plt.legend()
    img = Image(io.BytesIO(_fig_to_png_bytes(fig)))
    img.drawHeight = 7 * cm
    img.drawWidth = 16 * cm
    story.append(img)
    story.append(Spacer(1, 0.4 * cm))

    # Risk table
    rr = risk_row.copy()
    r = rr.reset_index()
    r.columns = ["Metric", "Value"]
    table_data = [r.columns.tolist()] + r.values.tolist()
    t = Table(table_data, hAlign="LEFT", colWidths=[7 * cm, 9 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(t)

    story.append(PageBreak())

    # 4) Technical explanation
    story.append(Paragraph("4. Technical indicators & how to use them", styles["Heading2"]))
    story.append(Paragraph(technical_rules_text.replace("\n", "<br/>"), styles["BodyText"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
