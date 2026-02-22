from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO


def generate_pdf_report(model_name, accuracy, report, confusion_matrix):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    text = c.beginText(40, 800)
    text.setFont("Helvetica", 11)

    text.textLine("Autism Prediction Report")
    text.textLine("-" * 40)
    text.textLine(f"Model Used: {model_name}")
    text.textLine(f"Accuracy: {accuracy * 100:.2f}%")
    text.textLine("")
    text.textLine("Classification Report:")
    text.textLine(report)
    text.textLine("")
    text.textLine("Confusion Matrix:")

    for row in confusion_matrix:
        text.textLine(str(row))

    c.drawText(text)
    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer.getvalue()
