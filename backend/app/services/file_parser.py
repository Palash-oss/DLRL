import pandas as pd
import io
from fastapi import UploadFile, HTTPException
from typing import List, Dict, Any
import PyPDF2

async def parse_file(file: UploadFile) -> List[Dict[str, Any]]:
    """
    Parse uploaded file (CSV, Excel, PDF) and return a list of records.
    Each record should ideally have a 'text' field.
    """
    filename = file.filename.lower()
    content = await file.read()
    
    data = []

    try:
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(io.BytesIO(content))
                data = _process_dataframe(df)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        elif filename.endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(io.BytesIO(content))
                data = _process_dataframe(df)
            except ImportError:
                raise HTTPException(status_code=500, detail="Server missing 'openpyxl' library required for Excel files.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid Excel file: {str(e)}")
            
        elif filename.endswith('.pdf'):
            try:
                text = _parse_pdf(content)
                if not text.strip():
                    raise ValueError("No text could be extracted from this PDF.")
                data = [{'text': text, 'source': 'pdf_full_content', 'company': 'Unknown'}]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV, Excel, or PDF.")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error parsing file: {str(e)}")
        
    return data

def _process_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to list of dicts, ensuring a 'text' column exists.
    """
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Look for potential text columns
    text_candidates = ['text', 'content', 'feedback', 'review', 'comment', 'description', 'message']
    text_col = None
    
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break
            
    if not text_col:
        # If no obvious text column, use the first string column that has an average length > 10
        # This avoids picking up ID columns or short codes
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check first few non-null values
                sample = df[col].dropna().head(5)
                if len(sample) > 0 and sample.str.len().mean() > 10:
                    text_col = col
                    break
        
        # Fallback: just take the first object column if nothing else
        if not text_col:
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_col = col
                    break
    
    if not text_col:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Could not find a text/content column in the file. Available columns: {available_cols}")
        
    # Rename found column to 'text' for consistency
    df = df.rename(columns={text_col: 'text'})
    
    # Replace NaN with empty string
    df = df.fillna('')
    
    return df.to_dict(orient='records')

def _parse_pdf(content: bytes) -> str:
    """
    Extract text from PDF bytes.
    """
    pdf_file = io.BytesIO(content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
