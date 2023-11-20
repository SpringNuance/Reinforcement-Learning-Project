import os
from pdf2image import convert_from_path

def convert_pdf_to_png(pdf_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            images = convert_from_path(pdf_path)

            # Save each page as an image
            for i, image in enumerate(images):
                image_name = f"{os.path.splitext(filename)[0]}.png"
                image_path = os.path.join(output_folder, image_name)
                image.save(image_path, "PNG")
                print(f"Saved: {image_path}")

pdf_folder = './'
output_folder = './'
convert_pdf_to_png(pdf_folder, output_folder)
