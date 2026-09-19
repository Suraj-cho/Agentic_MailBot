import imaplib
import email
from email.header import decode_header
import os

from dotenv import load_dotenv
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END

import smtplib
from email.message import EmailMessage
from email.utils import parseaddr

# =========================================================
# ENV
# =========================================================

load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")


# =========================================================
# GMAIL
# =========================================================

def get_unread_emails():

    imap = imaplib.IMAP4_SSL("imap.gmail.com")

    imap.login(
        EMAIL_ADDRESS,
        EMAIL_PASSWORD
    )

    imap.select("INBOX")

    status, messages = imap.search(
        None,
        "UNSEEN"
    )

    email_list = []

    email_ids = messages[0].split()

    print(f"Total unread emails: {len(email_ids)}")

    email_ids = email_ids[-5:]

    print(
        f"Processing latest {len(email_ids)} unread emails"
    )

    for email_id in email_ids:

        status, msg_data = imap.fetch(
            email_id,
            "(RFC822)"
        )

        raw_email = msg_data[0][1]

        msg = email.message_from_bytes(
            raw_email
        )

        # -------------------------
        # Subject
        # -------------------------

        subject = decode_header(
            msg["Subject"]
        )[0][0]

        if isinstance(subject, bytes):

            subject = subject.decode(
                errors="ignore"
            )

        # -------------------------
        # Sender
        # -------------------------

        sender = msg.get("From")

        # -------------------------
        # Body
        # -------------------------

        body = ""

        if msg.is_multipart():

            for part in msg.walk():

                if (
                    part.get_content_type()
                    == "text/plain"
                ):

                    payload = part.get_payload(
                        decode=True
                    )

                    if payload:

                        body = payload.decode(
                            errors="ignore"
                        )

                    break

        else:

            payload = msg.get_payload(
                decode=True
            )

            if payload:

                body = payload.decode(
                    errors="ignore"
                )

        email_data = {

            "subject": subject,

            "sender": sender,

            "body": body
        }

        email_list.append(
            email_data
        )

    imap.close()

    imap.logout()

    return email_list


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(

    model="openai/gpt-oss-120b",

    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

class EmailClassifier(BaseModel):

    category: Literal[
        "Type A",
        "Type B",
        "Other"
    ] = Field(
        description="Email category"
    )

    reason: str = Field(
        description="Reason for classification"
    )


structured_llm = llm.with_structured_output(
    EmailClassifier
)


# =========================================================
# STATE
# =========================================================



class EmailState(TypedDict):
    subject: str
    sender: str
    body: str
    category: str
    reason: str
    answer: str
# =========================================================
# CLASSIFIER NODE
# =========================================================

def classifier_node(
    state: EmailState
):

    subject = state["subject"]

    body = state["body"]

    body = body[:6000]

    prompt = f"""
You are an e-commerce customer support email classifier.

Classify the email into exactly one category.

Type A:
General questions that can be answered using FAQ,
product information, return policy, refund policy, etc.

Type B:
Questions that require customer-specific or
order-specific information from the database.

Other:
Emails that are not related to e-commerce
customer support.

Examples of Type A:

- What is your return policy?
- How can I get a refund?
- Tell me about this product.

Examples of Type B:

- Where is my order #1234?
- What is the status of order #5678?
- Cancel my order #9999.

Examples of Other:

- Job recruitment emails
- Bank notifications
- Security alerts
- Newsletters
- Personal emails

Subject:

{subject}

Email:

{body}
"""

    result = structured_llm.invoke(
        prompt
    )

    return {

        "category": result.category,

        "reason": result.reason
    }



# =========================================================
# RAG
# =========================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -------------------------
# Embedding Model
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# Create RAG Vector Store
# -------------------------

def create_vectorstore():

    # Read FAQ document
    with open(
        "faq.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    # Split document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.create_documents(
        [text]
    )


    # Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    return vectorstore        


def retrieve_information(question):

    vectorstore = create_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    documents = retriever.invoke(
        question
    )

    return documents



def generate_rag_answer(question):

    # Step 1: Retrieve relevant documents
    documents = retrieve_information(question)

    # Step 2: Convert documents into context
    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    # Step 3: Create prompt
    prompt = f"""
You are an e-commerce customer support assistant.

Answer the customer's question using ONLY the
information provided in the context.

If the answer is not available in the context,
say that you do not have enough information.

Context:
{context}

Customer Question:
{question}
"""

    # Step 4: Ask Groq
    response = llm.invoke(prompt)

    return response.content

def type_a_node(state: EmailState):

    question = state["body"]

    answer = generate_rag_answer(question)

    return {
        "answer": answer
    }

            

import sqlite3


def create_database():

    connection = sqlite3.connect(
        "ecommerce_workspace.db"
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            customer_name TEXT,
            customer_email TEXT,
            product_name TEXT,
            order_date TEXT,
            status TEXT,
            amount REAL
        )
    """)

    orders = [
        (1234, "Rahul", "rahul@gmail.com", "Shoes", "2026-09-01", "Shipped", 2999),
        (1235, "Amit", "amit@gmail.com", "Laptop", "2026-09-02", "Processing", 55000),
        (1236, "Priya", "priya@gmail.com", "T-Shirt", "2026-09-03", "Delivered", 999),
        (1237, "Sneha", "sneha@gmail.com", "Headphones", "2026-09-04", "Cancelled", 1999)
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO orders
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, orders)

    connection.commit()
    connection.close()

    print("Database created successfully.")        
 
    
import re

# ---------------------------

from langchain_community.utilities import SQLDatabase    
from langchain_community.agent_toolkits import create_sql_agent

db = SQLDatabase.from_uri(
    "sqlite:///ecommerce_workspace.db"
)

# sql_agent = create_sql_agent(
#     llm=llm,
#     db=db,
#     verbose=True
# )

def generate_sql(question):

    schema = db.get_table_info()

    prompt = f"""
You are a SQL expert.

Database schema:
{schema}

Write ONLY the SQL query needed to answer the customer's question.

Customer question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


def execute_sql(sql_query):
    
    sql_query = sql_query.replace("```sql", "")
    sql_query = sql_query.replace("```", "")
    sql_query = sql_query.strip()

    result = db.run(sql_query)

    return result



def generate_sql_answer(question, sql_result):

    prompt = f"""
You are an e-commerce customer support assistant.

Answer the customer's question using the database result.

Customer Question:
{question}

Database Result:
{sql_result}

Give a short, clear and professional answer.
"""

    response = llm.invoke(prompt)

    return response.content

def type_b_node(state: EmailState):

    question = state["body"]

    sql_query = generate_sql(question)

    sql_result = execute_sql(sql_query)

    answer = generate_sql_answer(
        question,
        sql_result
    )

    return {
        "answer": answer
    }    

def router(state: EmailState):

    if state["category"] == "Type A":
        return "type_a"

    if state["category"] == "Type B":
        return "type_b"

    return "end"

def send_email_node(state: EmailState):

    sender = state["sender"]
    subject = state["subject"]
    answer = state["answer"]

    # Extract only email address
    sender_email = parseaddr(sender)[1]

    msg = EmailMessage()

    msg["From"] = os.getenv("EMAIL_ADDRESS")
    msg["To"] = sender_email
    msg["Subject"] = "Re: " + subject

    msg.set_content(answer)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:

        smtp.login(
            os.getenv("EMAIL_ADDRESS"),
            os.getenv("EMAIL_PASSWORD")
        )

        smtp.send_message(msg)

    print("Reply sent successfully to:", sender_email)

    return {}
# =========================================================
# LANGGRAPH
# =========================================================

graph = StateGraph(
    EmailState
)


graph.add_node("classifier",classifier_node)

graph.add_node("type_a",type_a_node)

graph.add_node("type_b", type_b_node)

graph.add_node("send_email", send_email_node)

graph.add_conditional_edges(
    "classifier",
    router,
    {
        "type_a": "type_a",
        "type_b": "type_b",
        "end": END
    }
)

graph.add_edge(START,"classifier")

graph.add_edge("type_a", "send_email")
graph.add_edge("type_b", "send_email")

graph.add_edge("send_email", END)


app = graph.compile()

# -------------------------------------------------

if __name__ == "__main__":

    emails = get_unread_emails()

    for i, email_data in enumerate(emails, 1):

        initial_state = {
            "subject": email_data["subject"],
            "sender": email_data["sender"],
            "body": email_data["body"],
            "category": "",
            "reason": "",
            "answer": ""
        }

        result = app.invoke(initial_state)

        print("\n==============================")
        print(f"EMAIL {i}")
        print("==============================")
        print("Subject:", result["subject"])
        print("Sender:", result["sender"])
        print("Category:", result["category"])
        print("Reason:", result["reason"])

        if result["category"] in ["Type A", "Type B"]:
            print("Answer:", result["answer"])