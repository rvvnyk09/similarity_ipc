import PyPDF2

def extract_text_from_pdf(file):
    pdf_file = open(file, 'rb')

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    num_of_pages = len(pdf_reader.pages)
    text = ''

    for page_number in range(num_of_pages):
        page = pdf_reader.pages[page_number]

        page_content = page.extract_text()

        text += page_content

    pdf_file.close()

    return text


text = extract_text_from_pdf('BNSS.pdf')

file = open('BNSS.txt', 'w')
file.write(text)
file.close()
