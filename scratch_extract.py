import docx
import PyPDF2

pdf_path = r"e:\Bengali-MultiDomain-LLM-Evaluation\Review\LLM hossen.pdf"
docx_path = r"e:\Bengali-MultiDomain-LLM-Evaluation\Review\Response to Reviewers.docx"

with open(r"e:\Bengali-MultiDomain-LLM-Evaluation\extracted_text.txt", "w", encoding="utf-8") as out_f:
    out_f.write("--- PDF CONTENTS ---\n")
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                out_f.write(page.extract_text() + "\n")
    except Exception as e:
        out_f.write(str(e) + "\n")

    out_f.write("\n--- DOCX CONTENTS ---\n")
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            out_f.write(para.text + "\n")
    except Exception as e:
        out_f.write(str(e) + "\n")
