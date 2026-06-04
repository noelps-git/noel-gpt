from pypdf import PdfReader

reader = PdfReader("training data for posts.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

with open("my_writing.txt", "w") as f:
    f.write(text)

print("Done. Total characters:", len(text))