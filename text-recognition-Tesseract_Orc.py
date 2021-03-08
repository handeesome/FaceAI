from PIL import Image
import pytesseract

path = "img\\eng.png"

text = pytesseract.image_to_string(Image.open(path), lang='eng')
print(text)