import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in tqdm(range(num_pages), desc=f"Extracting text from {os.path.basename(pdf_path)}"):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_html(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text

if __name__ == "__main__":
    pdf_files = [
        'data/raw/documentation1.pdf',
        'data/raw/documentation2.pdf'
    ]
    
    html_urls = [
        'https://www.tflexcad.ru/help/cad/15/m_1.htm',
        'https://www.tflexcad.ru/help/cad/15/m_2.htm',
        'https://www.tflexcad.ru/help/cad/15/m_3.htm',
        'https://www.tflexcad.ru/help/cad/15/m_4.htm',
        'https://www.tflexcad.ru/help/cad/15/m_5.htm',
        'https://www.tflexcad.ru/help/cad/15/m_6.htm',
        'https://www.w3schools.com/cs/index.php',
        'https://www.w3schools.com/cs/cs_intro.php',
        'https://www.w3schools.com/cs/cs_getstarted.php',
        'https://www.w3schools.com/cs/cs_syntax.php',
        'https://www.w3schools.com/cs/cs_output.php',
        'https://www.w3schools.com/cs/cs_comments.php',
        'https://www.w3schools.com/cs/cs_variables.php',
        'https://www.w3schools.com/cs/cs_variables_constants.php',
        'https://www.w3schools.com/cs/cs_variables_display.php',
        'https://www.w3schools.com/cs/cs_variables_multiple.php',
        'https://www.w3schools.com/cs/cs_variables_identifiers.php',
        'https://www.w3schools.com/cs/cs_data_types.php',
        'https://www.w3schools.com/cs/cs_type_casting.php',
        'https://www.w3schools.com/cs/cs_user_input.php',
        'https://www.w3schools.com/cs/cs_operators.php',
        'https://www.w3schools.com/cs/cs_operators_assignment.php',
        'https://www.w3schools.com/cs/cs_operators_comparison.php',
        'https://www.w3schools.com/cs/cs_operators_logical.php',
        'https://www.w3schools.com/cs/cs_math.php',
        'https://www.w3schools.com/cs/cs_strings.php',
        'https://www.w3schools.com/cs/cs_strings_concat.php',
        'https://www.w3schools.com/cs/cs_strings_interpol.php',
        'https://www.w3schools.com/cs/cs_strings_access.php',
        'https://www.w3schools.com/cs/cs_strings_chars.php',
        'https://www.w3schools.com/cs/cs_booleans.php',
        'https://www.w3schools.com/cs/cs_conditions.php',
        'https://www.w3schools.com/cs/cs_conditions_else.php',
        'https://www.w3schools.com/cs/cs_conditions_elseif.php',
        'https://www.w3schools.com/cs/cs_conditions_shorthand.php',
        'https://www.w3schools.com/cs/cs_switch.php',
        'https://www.w3schools.com/cs/cs_while_loop.php',
        'https://www.w3schools.com/cs/cs_for_loop.php',
        'https://www.w3schools.com/cs/cs_foreach_loop.php',
        'https://www.w3schools.com/cs/cs_break.php',
        'https://www.w3schools.com/cs/cs_arrays.php',
        'https://www.w3schools.com/cs/cs_arrays_loop.php',
        'https://www.w3schools.com/cs/cs_arrays_sort.php',
        'https://www.w3schools.com/cs/cs_arrays_multi.php',
        'https://www.w3schools.com/cs/cs_methods.php',
        'https://www.w3schools.com/cs/cs_method_parameters.php',
        'https://www.w3schools.com/cs/cs_method_parameters_default.php',
        'https://www.w3schools.com/cs/cs_method_parameters_return.php',
        'https://www.w3schools.com/cs/cs_method_parameters_named_args.php',
        'https://www.w3schools.com/cs/cs_method_overloading.php',
        'https://www.w3schools.com/cs/cs_oop.php',
        'https://www.w3schools.com/cs/cs_classes.php',
        'https://www.w3schools.com/cs/cs_classes_multi.php',
        'https://www.w3schools.com/cs/cs_class_members.php',
        'https://www.w3schools.com/cs/cs_constructors.php',
        'https://www.w3schools.com/cs/cs_access_modifiers.php',
        'https://www.w3schools.com/cs/cs_properties.php',
        'https://www.w3schools.com/cs/cs_inheritance.php',
        'https://www.w3schools.com/cs/cs_polymorphism.php',
        'https://www.w3schools.com/cs/cs_abstract.php',
        'https://www.w3schools.com/cs/cs_interface.php',
        'https://www.w3schools.com/cs/cs_interface_multi.php',
        'https://www.w3schools.com/cs/cs_enums.php',
        'https://www.w3schools.com/cs/cs_files.php',
        'https://www.w3schools.com/cs/cs_exceptions.php',
        'https://www.w3schools.com/cs/cs_howto_add_two_numbers.php',
        'https://www.w3schools.com/cs/cs_examples.php',
        'https://www.w3schools.com/cs/cs_compiler.php',
    ]

    combined_text = ""

    # Extract text from PDF files
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"Extracting text from {pdf_path}...")
            combined_text += extract_text_from_pdf(pdf_path) + "\n"
        else:
            print(f"PDF file {pdf_path} not found. Skipping PDF extraction.")

    # Extract text from HTML files
    for html_url in html_urls:
        print(f"Extracting text from {html_url}...")
        combined_text += extract_text_from_html(html_url) + "\n"

    # Append combined text to file
    print("Appending combined text to file...")
    with open('data/processed/combined_text.txt', 'a', encoding='utf-8') as file:
        file.write(combined_text)

    print("Done!")
