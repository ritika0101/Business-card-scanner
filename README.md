# Business Card Scanner (OCR + NER)

A Jupyter-based pipeline that **reads business cards**, **extracts text with Tesseract OCR**, then **labels key fields** (Name, Organization, Designation, Phone, Email, Website) using a **spaCy Named Entity Recognizer** trained from examples derived directly from OCR output. A lightweight **Gradio UI** returns an **annotated image** (bounding boxes + labels) and a **structured JSON** of the extracted fields.

---

## Highlights

- **OCR**: `pytesseract.image_to_data` for tokens + word-level boxes.
- **Text cleaning**: whitespace/punctuation stripping + normalization.
- **Auto-annotation builder**: converts token-level tags into **BIO spans** and prepares spaCy training data.
- **Custom NER (spaCy)**: trains a model on your auto-built dataset.
- **Post-processing parsers**: canonicalize `PHONE`, `EMAIL`, `WEB`, names, org, etc.
- **BBox grouping**: merges adjacent tokens of the same entity into a single card-level bounding box.
- **Gradio app**: upload a card → get **(1)** annotated image and **(2)** JSON of entities.

Entities used: `NAME`, `ORG`, `DES`, `PHONE`, `EMAIL`, `WEB` (code also reserves `ADD` in the output dict).

---

## Notebook Workflow

**1) OCR → DataFrame**  
- Reads an input card image  
- Converts TSV to a pandas DataFrame with useful columns:  
  `text`, `left`, `top`, `width`, `height`, `conf` (confidence)  
- Cleaning:  
  Lowercases   
  Removes string.whitespace & punctuation via translation tables  
  Function: `clean_text(txt)` applied to the text column.  
- Filters empty strings → `dataClean`.  

**2) Group by Card & Build BIO Annotations**  
- `dataClean.groupby("id")` to process each card (e.g., "1.jpg", "2.jpg").  
- For each card:  
  Takes [["text","tag"]] where `tag` is token-level label like `B-NAME`, `I-ORG`, or `O`.  
- Builds spaCy span annotations:  
  Concatenates tokens with a trailing space; tracks character offsets (start, end).  
  For any non-O label, appends (start, end-1, LABEL) to "entities".  
- Appends (content_string, {"entities": [...]}) to dataset.  

Note: You should ensure your input DataFrame has a tag column per token using BIO tags (e.g., B-NAME, I-NAME, O). The notebook assumes such tags exist (from prior manual labeling or a CSV).  

**3) Train spaCy NER (blank English)**  

**4) Inference Pipeline**
`parser(text, label)`   
- Normalizes tokens by field:  
  PHONE: strip non-digits.   
  EMAIL: allow @_.- and alphanumerics.  
  WEB: allow URL punctuation :/.%#-.  
  NAME/DES: alpha-only with spaces, then title-case.  
  ORG: alphanumerics + spaces, then title-case.  

`GroupGen (merge tool)`  
Keeps a running group id so adjacent tokens with same entity label map to the same group:

`getPredictions(image) -> (PIL.Image, dict)`  
1. Run Tesseract image_to_data → DataFrame.  
2. Clean text & drop empties.  
3. Token align: builds start/end per cleaned token and merges with doc.to_json() tokens from model_ner(content).  
4. Keep only non-O labels → convert B-NAME/I-NAME → NAME etc.  
5. Group & bbox:  
- For each group, compute min/max over left, top, right, bottom.  
- Draw rectangles and labels via cv2.rectangle and cv2.putText.  
6. Aggregate entities:   
- BIO logic stitches multi-token fields:  
  - For NAME, ORG, DES: join with spaces.  
  - For PHONE, EMAIL, WEB: concatenate (no extra space), then parser cleans them.  
7. Returns:   
- Annotated image (PIL.Image.fromarray(img_bb))  
- Entities dict: {"NAME":[], "ORG":[], "DES":[], "PHONE":[], "EMAIL":[], "WEB":[], "ADD":[]}  
