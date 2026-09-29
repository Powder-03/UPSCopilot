"""Ingestion script for Key Central Statutes using LangChain Document and Chroma.
Covers critical sections of RPA 1951, RTI 2005, Lokpal 2013, CVC Act, PMLA, and DPDP 2023.
"""
import logging
from typing import List
from langchain_core.documents import Document
from src.kb.vector_store import get_chroma_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CENTRAL_ACT_CHUNKS = [
    {
        "act_name": "Representation of the People Act, 1951",
        "act_slug": "rpa1951",
        "section_num": "8",
        "title": "Disqualification on conviction for certain offences",
        "text": "A person convicted of an offence punishable under specified sections (including promoting enmity, bribery, rape, hoarding) or sentenced to imprisonment for not less than two years shall be disqualified from the date of such conviction and shall continue to be disqualified for a further period of six years since his release. In Lily Thomas v. Union of India (2013), the Supreme Court struck down Section 8(4) which previously granted sitting MPs/MLAs a 3-month protection window.",
        "citation": "Section 8, Representation of the People Act, 1951",
    },
    {
        "act_name": "Representation of the People Act, 1951",
        "act_slug": "rpa1951",
        "section_num": "29A",
        "title": "Registration with the Election Commission of associations and bodies as political parties",
        "text": "Any association or body of individual citizens calling itself a political party must apply to the Election Commission for registration. The application must bear a specific undertaking affirming true faith and allegiance to the Constitution of India and principles of socialism, secularism and democracy.",
        "citation": "Section 29A, Representation of the People Act, 1951",
    },
    {
        "act_name": "Representation of the People Act, 1951",
        "act_slug": "rpa1951",
        "section_num": "123",
        "title": "Corrupt practices",
        "text": "Defines corrupt electoral practices including bribery, undue influence with electoral rights, appeal to vote or refrain from voting on grounds of religion, race, caste, community or language, publication of false statements, and hiring of vehicles for electors. Abhiram Singh v. C.D. Commachen (2017) 7-judge bench held that 'his religion' extends to candidate, voter, or agent.",
        "citation": "Section 123, Representation of the People Act, 1951",
    },
    {
        "act_name": "Right to Information Act, 2005",
        "act_slug": "rti2005",
        "section_num": "4",
        "title": "Obligations of public authorities (Suo Motu Proactive Disclosure)",
        "text": "Every public authority shall maintain all its records duly catalogued and indexed, and publish within 120 days particulars of its organization, functions, duties, powers of officers, decision making procedures, norms set for discharge of functions, budget allocated, and subsidy programmes. Non-compliance is the root cause of RTI appeal backlogs.",
        "citation": "Section 4, Right to Information Act, 2005",
    },
    {
        "act_name": "Right to Information Act, 2005",
        "act_slug": "rti2005",
        "section_num": "8",
        "title": "Exemption from disclosure of information",
        "text": "Exempts information disclosure affecting sovereignty, integrity, security, scientific or economic interests of the State; information expressly forbidden by court; commercial confidence, trade secrets or IP; information held in fiduciary relationship; and personal information having no relationship to public activity unless public interest outweighs privacy.",
        "citation": "Section 8, Right to Information Act, 2005",
    },
    {
        "act_name": "Lokpal and Lokayuktas Act, 2013",
        "act_slug": "lokpal2013",
        "section_num": "14",
        "title": "Jurisdiction of Lokpal over public functionaries",
        "text": "Lokpal has statutory jurisdiction to inquire into allegations of corruption against the Prime Minister (with specific national security safeguards), Ministers of the Union, Members of Parliament, and Group A, B, C, and D civil servants. Mandated by 2nd Administrative Reforms Commission.",
        "citation": "Section 14, Lokpal and Lokayuktas Act, 2013",
    },
    {
        "act_name": "Central Vigilance Commission Act, 2003",
        "act_slug": "cvc2003",
        "section_num": "8",
        "title": "Functions and powers of Central Vigilance Commission",
        "text": "The Commission shall exercise superintendence over the functioning of the Delhi Special Police Establishment (CBI) insofar as it relates to investigation of offences under the Prevention of Corruption Act, 1988. Gives directions to DSPE and reviews probe progress.",
        "citation": "Section 8, Central Vigilance Commission Act, 2003",
    },
    {
        "act_name": "Prevention of Money Laundering Act, 2002",
        "act_slug": "pmla2002",
        "section_num": "45",
        "title": "Offences to be cognizable and non-bailable (Twin Conditions for Bail)",
        "text": "Bail shall not be granted to a person accused of an offence under PMLA unless the Public Prosecutor has been given opportunity to oppose, and the Court is satisfied that there are reasonable grounds for believing the accused is NOT guilty and not likely to commit offence while on bail (Vijay Madanlal Choudhary v. UOI, 2022).",
        "citation": "Section 45, Prevention of Money Laundering Act, 2002",
    },
    {
        "act_name": "Digital Personal Data Protection Act, 2023",
        "act_slug": "dpdp2023",
        "section_num": "8",
        "title": "General obligations of Data Fiduciary",
        "text": "A Data Fiduciary shall make reasonable security safeguards to prevent personal data breach, ensure accuracy and completeness of personal data, erase personal data upon withdrawal of consent, and establish an effective grievance redressal mechanism.",
        "citation": "Section 8, Digital Personal Data Protection Act, 2023",
    },
]


def load_central_act_documents() -> List[Document]:
    """Converts central statutory provisions into standard LangChain Document objects."""
    docs = []
    for item in CENTRAL_ACT_CHUNKS:
        full_title = f"{item['act_name']} - Section {item['section_num']}: {item['title']}"
        doc = Document(
            page_content=f"{full_title}\n\n{item['text']}",
            metadata={
                "id": f"{item['act_slug']}_sec_{item['section_num']}",
                "act_name": item["act_name"],
                "section": item["section_num"],
                "title": full_title,
                "doc_type": "statute",
                "citation": item["citation"],
                "gs_paper": "GS2",
            },
        )
        docs.append(doc)
    return docs


def ingest_central_acts(persist_dir: str = None) -> List[Document]:
    """Ingests central statute documents into Chroma vector store."""
    docs = load_central_act_documents()
    vector_store = get_chroma_vector_store(persist_dir)
    vector_store.add_documents(docs)
    logger.info(f"Successfully added {len(docs)} central statutory sections to Chroma vector store.")
    return docs


if __name__ == "__main__":
    ingest_central_acts()
