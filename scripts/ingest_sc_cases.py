"""Ingestion script for Landmark Supreme Court Jurisprudence using LangChain.
Structures landmark cases into ratio decidendi and operative guidelines.
"""
import logging
from typing import List
from langchain_core.documents import Document
from src.kb.vector_store import get_chroma_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANDMARK_CASES = [
    {
        "case_name": "Kesavananda Bharati v. State of Kerala",
        "citation": "(1973) 4 SCC 225",
        "year": 1973,
        "bench_strength": 13,
        "ratio": "The largest 13-judge Constitution Bench held by a 7-6 majority that while Parliament has wide power to amend any part of the Constitution under Article 368 including Fundamental Rights, it cannot alter, destroy or abrogate the 'Basic Structure' or essential identity of the Constitution.",
        "guidelines": "Core basic structure features include: Supremacy of the Constitution, Republican & Democratic form of governance, Secularism, Separation of powers, and Federalism.",
    },
    {
        "case_name": "S.R. Bommai v. Union of India",
        "citation": "(1994) 3 SCC 1",
        "year": 1994,
        "bench_strength": 9,
        "ratio": "A 9-judge Constitution Bench ruled that federalism and secularism are foundational parts of the Basic Structure. Proclamation of President's Rule under Article 356 is subject to judicial review. The floor of the legislative assembly is the ONLY constitutional forum to test a government's majority, not subjective Raj Bhavan opinions.",
        "guidelines": "State assembly cannot be dissolved until both Houses of Parliament approve Article 356 proclamation. Courts can revive wrongfully dissolved assemblies.",
    },
    {
        "case_name": "Justice K.S. Puttaswamy (Retd.) v. Union of India",
        "citation": "(2017) 10 SCC 1",
        "year": 2017,
        "bench_strength": 9,
        "ratio": "A unanimous 9-judge Constitution Bench held that the Right to Privacy is a fundamental and inalienable right intrinsic to life and personal liberty under Article 21, overruling M.P. Sharma (1954) and Kharak Singh (1962). Encompasses informational privacy, bodily autonomy, and spatial privacy.",
        "guidelines": "Any state restriction on privacy must satisfy the 3-fold proportionality test: (1) Legality (valid law), (2) Legitimate state goal, and (3) Proportionality (least restrictive means).",
    },
    {
        "case_name": "Maneka Gandhi v. Union of India",
        "citation": "(1978) 1 SCC 248",
        "year": 1978,
        "bench_strength": 7,
        "ratio": "Overruled A.K. Gopalan. Ruled that 'procedure established by law' under Article 21 cannot be arbitrary, oppressive, or fanciful; it must be 'just, fair, and reasonable', infusing American substantive 'due process' into Indian constitutional law. Established the Golden Triangle connecting Articles 14, 19, and 21.",
        "guidelines": "Procedure depriving personal liberty must strictly conform with natural justice (audi alteram partem).",
    },
    {
        "case_name": "Association for Democratic Reforms (ADR) v. Union of India (Electoral Bonds Case)",
        "citation": "2024 INSC 113",
        "year": 2024,
        "bench_strength": 5,
        "ratio": "A unanimous 5-judge Constitution Bench struck down the Electoral Bond Scheme and amendments to the Companies Act, RPA 1951, and Income Tax Act as unconstitutional. Held that voters have a fundamental Right to Information under Article 19(1)(a) regarding political funding to cast an informed vote.",
        "guidelines": "Directed State Bank of India to cease issuance and disclose complete purchaser and redemption details to Election Commission for public publication.",
    },
    {
        "case_name": "Lily Thomas v. Union of India",
        "citation": "(2013) 7 SCC 653",
        "year": 2013,
        "bench_strength": 2,
        "ratio": "Struck down Section 8(4) of the Representation of the People Act, 1951 as unconstitutional. Ruled that Parliament has no power under Articles 102(1)(e) and 191(1)(e) to create a protective 3-month moratorium shielding sitting legislators from instant disqualification upon conviction.",
        "guidelines": "Disqualification takes effect immediately on the date of conviction; sitting members cannot claim differential immunity.",
    },
    {
        "case_name": "Govt of NCT of Delhi v. Union of India (Services Case)",
        "citation": "(2023) 9 SCC 1",
        "year": 2023,
        "bench_strength": 5,
        "ratio": "A 5-judge Constitution Bench held that the elected government of NCT Delhi exercises legislative and executive control over civil services ('Services' under Entry 41 of State List), excluding Police, Public Order, and Land. Articulated the 'Triple Chain of Accountability': civil servants are accountable to ministers, ministers are accountable to legislature, and legislature is accountable to citizens.",
        "guidelines": "Lieutenant Governor is bound by the aid and advise of the Council of Ministers on all matters within legislative competence.",
    },
    {
        "case_name": "State of Punjab v. Principal Secretary to Governor",
        "citation": "2023 INSC 1018",
        "year": 2023,
        "bench_strength": 3,
        "ratio": "Ruled that the Governor cannot sit indefinitely on bills passed by the State Legislature or exercise a pocket veto. Under Article 200, if the Governor withholds assent, he MUST return the bill to the Assembly 'as soon as possible' with a message. The Governor cannot derail state legislative democracy.",
        "guidelines": "If the State Legislature passes the bill again with or without amendments, the Governor HAS NO OPTION but to grant assent.",
    },
    {
        "case_name": "Union of India v. Mohit Minerals Pvt Ltd",
        "citation": "(2022) 10 SCC 700",
        "year": 2022,
        "bench_strength": 3,
        "ratio": "The Supreme Court ruled that recommendations of the GST Council under Article 279A are not binding on Parliament or State Legislatures, but have persuasive value. Highlighted that Indian federalism is a dialogue between two sovereign entities and collaborative fiscal federalism requires consensus, not coercion.",
        "guidelines": "Article 246A treats Parliament and State Legislatures as equal legislative authorities with simultaneous taxing power.",
    },
    {
        "case_name": "Anuradha Bhasin v. Union of India",
        "citation": "(2020) 3 SCC 637",
        "year": 2020,
        "bench_strength": 3,
        "ratio": "Ruled that freedom of speech and expression and freedom to practice any trade or profession over the internet are constitutionally protected under Article 19(1)(a) and 19(1)(g). Indefinite internet shutdowns are impermissible and suspension orders must satisfy the proportionality test.",
        "guidelines": "All internet suspension orders must be published publicly and are subject to mandatory periodic review by the Review Committee under Telecom Suspension Rules.",
    },
]


def load_sc_case_documents() -> List[Document]:
    """Converts landmark Supreme Court cases into standard LangChain Document objects."""
    docs = []
    for item in LANDMARK_CASES:
        content = f"{item['case_name']} ({item['year']}) - Ratio Decidendi\n\n{item['ratio']}"
        if item.get("guidelines"):
            content += f"\n\nOperative Guidelines & Doctrines:\n{item['guidelines']}"

        slug = item["case_name"].lower().replace("v.", "").replace(".", "").split()[0]
        doc = Document(
            page_content=content,
            metadata={
                "id": f"case_{slug}_{item['year']}",
                "case_name": item["case_name"],
                "citation": item["citation"],
                "year": item["year"],
                "bench_strength": item["bench_strength"],
                "title": f"{item['case_name']} ({item['year']})",
                "doc_type": "case_law",
                "gs_paper": "GS2",
            },
        )
        docs.append(doc)
    return docs


def ingest_sc_cases(persist_dir: str = None) -> List[Document]:
    """Ingests landmark Supreme Court case documents into Chroma vector store."""
    docs = load_sc_case_documents()
    vector_store = get_chroma_vector_store(persist_dir)
    vector_store.add_documents(docs)
    logger.info(f"Successfully added {len(docs)} landmark SC cases to Chroma vector store.")
    return docs


if __name__ == "__main__":
    ingest_sc_cases()
