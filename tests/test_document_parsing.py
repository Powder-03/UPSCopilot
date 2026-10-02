"""Offline unit tests for the document parsing and QCAB segmentation pipeline."""
from PIL import Image
from src.models.parsing import ParsedDocument, ParsedQuestion
from src.parsing.preprocessor import PDFPreprocessor
from src.parsing.segmenter import QCABSegmenter


def test_qcab_segmenter_calculates_correct_slices():
    """Verifies that QCABSegmenter slices standard 55-page booklet into 20 questions."""
    slices = QCABSegmenter.calculate_default_qcab_page_slices(total_pages=55)

    assert len(slices) == 20
    # Q1 to Q10 should have 2 pages each
    for s in slices[:10]:
        assert s["max_marks"] == 10.0
        assert len(s["pages"]) == 2

    # Q11 to Q20 should have 3 pages each
    for s in slices[10:]:
        assert s["max_marks"] == 15.0
        assert len(s["pages"]) == 3

    # Check boundaries (55-page FLTs start on page 3 after cover and evaluation rubric)
    assert slices[0]["q_num"] == 1
    assert slices[0]["pages"] == [3, 4]
    assert slices[19]["q_num"] == 20


def test_qcab_segmenter_reconciles_missing_questions():
    """Verifies that missing questions are automatically injected with blank answers."""
    # Candidate only attempted Q1, Q2, and Q4 (skipped Q3)
    extracted = [
        ParsedQuestion(q_num=1, max_marks=10.0, question="Q1 prompt", candidate_answer="Answer 1"),
        ParsedQuestion(q_num=4, max_marks=10.0, question="Q4 prompt", candidate_answer="Answer 4"),
        ParsedQuestion(q_num=2, max_marks=10.0, question="Q2 prompt", candidate_answer="Answer 2"),
    ]

    master = [
        {"q_num": 1, "max_marks": 10.0, "question": "Q1 prompt"},
        {"q_num": 2, "max_marks": 10.0, "question": "Q2 prompt"},
        {"q_num": 3, "max_marks": 10.0, "question": "Q3 prompt"},
        {"q_num": 4, "max_marks": 10.0, "question": "Q4 prompt"},
    ]

    reconciled = QCABSegmenter.reconcile_and_sort_questions(extracted, master_questions=master)

    assert len(reconciled) == 4
    # Should be sorted 1 to 4
    assert [q.q_num for q in reconciled] == [1, 2, 3, 4]

    # Q3 should be automatically injected as unattempted blank
    q3 = reconciled[2]
    assert q3.q_num == 3
    assert q3.is_blank is True
    assert q3.candidate_answer == ""
    assert q3.word_count == 0


def test_parsed_question_and_document_serialization():
    """Verifies schema conversion from ParsedDocument to evaluation engine JSON dicts."""
    q1 = ParsedQuestion(
        q_num=1,
        max_marks=10.0,
        question="Constitutional morality...",
        candidate_answer="Sample answer with [Diagram: Flowchart]",
        diagrams=["Flowchart"],
        word_count=50,
    )
    doc = ParsedDocument(
        source_file="test_copy.pdf",
        total_pages=2,
        questions=[q1],
    )

    eval_list = doc.to_evaluation_list()
    assert len(eval_list) == 1
    assert eval_list[0] == {
        "q_num": 1,
        "max_marks": 10.0,
        "question": "Constitutional morality...",
        "candidate_answer": "Sample answer with [Diagram: Flowchart]",
    }


def test_preprocessor_blank_page_heuristic():
    """Verifies that purely white or nearly-white images are detected as visually blank."""
    preprocessor = PDFPreprocessor(dpi=150)

    # Pure white image
    white_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    assert preprocessor.is_page_visually_blank(white_img) is True

    # Image with substantial dark ink (black text area)
    dark_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    for x in range(50, 150):
        for y in range(50, 150):
            dark_img.putpixel((x, y), (0, 0, 0))

    assert preprocessor.is_page_visually_blank(dark_img) is False


def test_clean_question_text_removes_hindi_and_metadata():
    """Verifies that _clean_question_text strips Devanagari script, numbering, marks, and word limits."""
    from src.parsing.pipeline import _clean_question_text

    # Bilingual prompt with Hindi first line, English second line, marks and word counts
    raw_bilingual = (
        "1. संघ लोक सेवा आयोग की स्वतंत्रता को बनाए रखने के उपायों की विवेचना कीजिए।\n"
        "Discuss the measures to preserve the independence of the Union Public Service Commission. "
        "(10 Marks, 150 words)"
    )
    cleaned = _clean_question_text(raw_bilingual)
    assert cleaned == "Discuss the measures to preserve the independence of the Union Public Service Commission."

    # Question with Q2: prefix and trailing marks
    raw_prefixed = "Q.2: What is judicial review? Discuss its constitutional foundations. (15 Marks)"
    cleaned_prefixed = _clean_question_text(raw_prefixed)
    assert cleaned_prefixed == "What is judicial review? Discuss its constitutional foundations."

    # Inline bilingual string
    raw_inline = "कुछ हिंदी शब्द Evaluate the effectiveness of the Sevottam model. (10 marks)"
    cleaned_inline = _clean_question_text(raw_inline)
    assert cleaned_inline == "Evaluate the effectiveness of the Sevottam model."

