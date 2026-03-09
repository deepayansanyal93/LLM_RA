"""
PDF file validation for the ingestion pipeline.

Validates that an input file exists, has the correct extension, and contains
valid PDF content (magic bytes). Used by text and image extractors before
attempting to open and parse the file.
"""

from pathlib import Path


class PDFValidationError(Exception):
    """
    Raised when the input file fails PDF validation.

    The error message describing the validation failure is stored in the
    exception and can be printed or accessed via str(exception).
    """

    def __init__(self, message: str = "PDF validation failed") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


def validate_pdf_file(file_path: str | Path) -> None:
    """
    Validate that the given path points to a valid PDF file.

    Performs the following checks:
    1. Path is not None or empty
    2. Path exists on the filesystem
    3. File has a .pdf extension (case-insensitive)
    4. File starts with PDF magic bytes (%PDF)

    Raises:
        PDFValidationError: If any validation check fails, with a descriptive message.

    Returns:
        None on success. Validation passes by not raising.
    """
    # Step 1: Ensure file_path is provided and non-empty
    if file_path is None:
        raise PDFValidationError("File path cannot be None.")

    path = Path(file_path)
    path_str = str(path).strip()

    if not path_str:
        raise PDFValidationError("File path cannot be empty.")

    # Step 2: Resolve to absolute path and check that the file exists
    resolved_path = path.resolve()

    if not resolved_path.exists():
        raise PDFValidationError(f"File does not exist: {resolved_path}")

    if not resolved_path.is_file():
        raise PDFValidationError(f"Path is not a file: {resolved_path}")

    # Step 3: Verify the file has a .pdf extension
    if resolved_path.suffix.lower() != ".pdf":
        raise PDFValidationError(
            f"File must have .pdf extension. Got: {resolved_path.suffix}"
        )

    # Step 4: Read first 8 bytes and verify PDF magic bytes
    # PDF files must start with "%PDF" per the PDF specification
    try:
        with open(resolved_path, "rb") as f:
            header = f.read(8)
    except OSError as e:
        raise PDFValidationError(f"Cannot read file: {e}") from e

    if not header.startswith(b"%PDF"):
        raise PDFValidationError(
            "File does not appear to be a valid PDF (missing %PDF header). "
            "The file may have been renamed or is corrupted."
        )
